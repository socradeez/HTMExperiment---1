import cupy as cp
from cupy import sparse

class FFProximalConnection:

    syn_perm_active_inc = 0.1
    syn_perm_inactive_dec = 0.03

    def __init__(self, parent_layer, input_layer, connected_perm=0.5, initial_perm_range=0.3, boost_strength=1, stim_threshold=3, boosting=True):

        self.perm_sparsity = 0.5
        self.parent_layer = parent_layer
        self.input_layer = input_layer
        self.connected_perm = connected_perm
        self.initial_perm_range = initial_perm_range
        self.stimulus_threshold = stim_threshold
        self.boosting = boosting
        self.active_duty_cycle = cp.zeros((self.num_dendrites, ))
        self.overlap_duty_cycle = cp.zeros((self.num_dendrites, ))
        self.boosting_values = cp.ones((self.num_dendrites, ))
        self.boost_strength = boost_strength
        self.step = 1
        self.permanences = self._initialize_permanences()

    @property
    def input_columns(self):
        return self.input_layer.num_columns
    
    @property
    def input_neurons_per_column(self):
        return self.input_layer.neurons_per_column
    
    @property
    def num_dendrites(self):
        return self.parent_layer.num_columns
    
    @property
    def connected_synapses(self):
        # Synapses are compared against the connection threshold to get the connected synapse matrix
        connected_synapses = self.permanences >= self.connected_perm
        return connected_synapses
    
    @property
    def overlaps(self):
        # Use tensordot to compute overlaps
        overlaps = cp.tensordot(self.input_layer.active_neurons.A.astype(int), self.connected_synapses, axes=([0, 1],[0, 1])).squeeze()
        if self.boosting:
            overlaps *= self.boosting_values
        # Returns an array of shape=(self.num_dendrites,)
        return overlaps

    def _initialize_permanences(self):
        # Generate a fully connected layer of permanences in a normal distribution centered over the connection threshold
        perms = cp.random.normal(self.connected_perm - self.initial_perm_range, 
                                 self.connected_perm + self.initial_perm_range, 
                                 (self.input_columns, self.input_neurons_per_column, self.num_dendrites))

        # Create a binary mask with the specified sparsity
        sparsity_mask = cp.random.rand(self.input_columns, self.input_neurons_per_column, self.num_dendrites) < self.perm_sparsity

        # Apply the sparsity mask to the perms matrix
        sparse_perms = perms * sparsity_mask

        return sparse_perms

    def compute_activity(self, inhibition):
        overlaps = self.overlaps
        if inhibition < 1:
            k = int(self.num_dendrites * inhibition)  # Desired number of active columns

            # Sort overlaps and find the kth largest value
            sorted_overlaps = cp.sort(overlaps)[::-1]
            kth_overlap = sorted_overlaps[max(0, k-1)]
            candidates = overlaps >= kth_overlap

            # Count the number of candidates
            num_candidates = cp.sum(candidates)

            if num_candidates > k:
                # If more than k candidates, select k of them at random
                candidate_indices = cp.where(candidates)[0]
                chosen_indices = cp.random.choice(candidate_indices, size=k, replace=False)
                self.activity = cp.zeros_like(overlaps, dtype=bool)
                self.activity[chosen_indices] = True
            else:
                # If k or fewer candidates, activate all of them
                self.activity = candidates

        else:
            self.activity = overlaps >= self.stimulus_threshold

        return self.activity

    def learn(self):
        overlaps = self.overlaps
        # Reshape self.activity to work with the 3D permanence array
        active_dendrites = self.activity.reshape(1, 1, -1)

        # Decrement all synapses of active dendrites
        # This applies the decrement to the entire slice for each active dendrite
        decrement_mask = active_dendrites * FFProximalConnection.syn_perm_inactive_dec
        self.permanences -= decrement_mask

        # Get active inputs reshaped to align with the 3D permanence array
        active_inputs = self.input_layer.active_neurons.A.reshape(self.input_columns, self.input_neurons_per_column, 1)

        # Increment synapses that connect active columns to active inputs
        increment_mask = active_dendrites * active_inputs * (FFProximalConnection.syn_perm_active_inc + FFProximalConnection.syn_perm_inactive_dec)
        self.permanences += increment_mask

        # Ensuring permanences stay within bounds
        self.permanences = cp.clip(self.permanences, 0, 1)

        self.update_active_duty_cycle()
        self.update_overlap_duty_cycle_and_adjust_perm(overlaps)
        self.update_boosting_values()

    def update_active_duty_cycle(self):
        num_timesteps = min(500, self.step)
        #activity = self.activity.astype(cp.float32)  # Convert activity to float for calculation
        self.active_duty_cycle = (self.active_duty_cycle * (num_timesteps - 1) + self.activity) / num_timesteps

    def update_overlap_duty_cycle_and_adjust_perm(self, overlaps):
        min_duty_cycle_fraction = 0.01
        num_timesteps = min(500, self.step)
        current_overlaps = overlaps > self.stimulus_threshold
        self.overlap_duty_cycle = (self.overlap_duty_cycle * (num_timesteps - 1) + current_overlaps) / num_timesteps

        # Adjust permanences if necessary
        max_duty_cycle = cp.max(self.overlap_duty_cycle)
        min_duty_cycle = min_duty_cycle_fraction * max_duty_cycle
        columns_to_increase = self.overlap_duty_cycle < min_duty_cycle
        columns_to_increase_reshaped = columns_to_increase.reshape(1, 1, -1)

        increase_amount = 0.1 * self.connected_perm

        # Create a broadcasted mask that matches the shape of self.permanences
        broadcasted_mask = cp.broadcast_to(columns_to_increase_reshaped, self.permanences.shape)

        # Apply the increase to the permanences where columns need to be increased
        self.permanences[broadcasted_mask] += increase_amount

        # Ensuring permanences stay within bounds
        self.permanences = cp.clip(self.permanences, 0, 1)

    def boost_function(self):
        mean_neighbors_duty_cycle = cp.mean(self.active_duty_cycle)
        return cp.exp(self.boost_strength * (mean_neighbors_duty_cycle - self.active_duty_cycle))

    def update_boosting_values(self):
        self.boosting_values = self.boost_function()