import cupy as cp
from cupy import sparse

class FFProximalConnection:

    syn_perm_active_inc = 0.1
    syn_perm_inactive_dec = 0.03

    def __init__(self, parent_layer, input_layer, connected_perm=0.5, initial_perm_range=0.2, boost_strength=1, stim_threshold=3):

        self.parent_layer = parent_layer
        self.input_layer = input_layer
        self.input_columns = input_layer.num_columns
        self.input_neurons_per_column = input_layer.neurons_per_column
        self.num_dendrites = parent_layer.num_columns
        self.connected_perm = connected_perm
        self.initial_perm_range = initial_perm_range
        self.stimulus_threshold = stim_threshold
        self.active_duty_cycle = cp.zeros((self.num_dendrites, ))
        self.overlap_duty_cycle = cp.zeros((self.num_dendrites, ))
        self.boosting_values = cp.ones((self.num_dendrites, ))
        self.boost_strength = boost_strength
        self.step = 1
        self.permanences = self._initialize_permanences()
        self.get_connected_synapses()

    def _initialize_permanences(self):

        perms = cp.random.normal(self.connected_perm - self.initial_perm_range, self.connected_perm + self.initial_perm_range, (self.input_columns, self.input_neurons_per_column, self.num_dendrites))
        return perms
    
    def get_activity(self, inhibition=1, boosting=True):
        """
        Runs a simulation step.

        Parameters:
            active_indices (list): Indices of active neurons in the neuron layer.

        Returns:
            list: Indices of the active columns after the step.
        """
        # Compute overlap scores for each column based on the current input
        self.compute_overlaps(boosting)

        # Determine the active columns based on the overlap scores and sparsity constraint
        return self.compute_activity(inhibition)
    
    def get_connected_synapses(self):
        # Generate a binary mask array representing the connected synapses (with perm value greater than or equal to self.connected_perm)
        self.connected_synapses = self.permanences >= self.connected_perm
        return self.connected_synapses

    def compute_overlaps(self, boosting):
        """
        Computes overlaps using tensordot between a 3D and a 2D matrix.
        """
        # Use tensordot to compute overlaps
        self.overlaps = cp.tensordot(self.input_layer.active_neurons.A.astype(int), self.connected_synapses, axes=([0, 1],[0, 1])).squeeze()

        # Multiply overlaps by boost factors if boosting is turned on
        if boosting:
            self.overlaps * self.boosting_values
        # Returns an array of shape=(self.num_dendrites,)
        return self.overlaps

    def compute_activity(self, inhibition):
        if inhibition < 1:
            k = int(self.num_dendrites * inhibition)  # Desired number of active columns

            # Sort overlaps and find the kth largest value
            sorted_overlaps = cp.sort(self.overlaps)[::-1]
            kth_overlap = sorted_overlaps[max(0, k-1)]
            candidates = self.overlaps >= kth_overlap

            # Count the number of candidates
            num_candidates = cp.sum(candidates)

            if num_candidates > k:
                # If more than k candidates, select k of them at random
                candidate_indices = cp.where(candidates)[0]
                chosen_indices = cp.random.choice(candidate_indices, size=k, replace=False)
                self.activity = cp.zeros_like(self.overlaps, dtype=bool)
                self.activity[chosen_indices] = True
            else:
                # If k or fewer candidates, activate all of them
                self.activity = candidates

        else:
            self.activity = self.overlaps >= self.stimulus_threshold

        return self.activity

    def learn(self):
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
        self.update_overlap_duty_cycle_and_adjust_perm()
        self.update_boosting_values()

    def update_active_duty_cycle(self):

        num_timesteps = min(500, self.step)
        #activity = self.activity.astype(cp.float32)  # Convert activity to float for calculation
        self.active_duty_cycle = (self.active_duty_cycle * (num_timesteps - 1) + self.activity) / num_timesteps

    def update_overlap_duty_cycle_and_adjust_perm(self):
        min_duty_cycle_fraction = 0.01
        num_timesteps = min(500, self.step)
        current_overlaps = self.overlaps > self.stimulus_threshold
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
        """
        Boosting function for adjusting the active duty cycle.

        Parameters:
            active_duty_cycle (cp.ndarray): The active duty cycle of columns.

        Returns:
            cp.ndarray: The boosted values.
        """
        mean_neighbors_duty_cycle = cp.mean(self.active_duty_cycle)
        return cp.exp(self.boost_strength * (mean_neighbors_duty_cycle - self.active_duty_cycle))

    def update_boosting_values(self):
        """
        Updates the boosting values for all columns.
        """
        self.boosting_values = self.boost_function()

    
