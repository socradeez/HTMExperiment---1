import numpy as np

import numpy as np

class FFProximalConnection:
    """
    Represents a FeedForward Proximal Connection layer in a Hierarchical Temporal Memory (HTM) system.

    Attributes:
        syn_perm_active_inc (float): Increment of synapse permanence for active connections.
        syn_perm_inactive_dec (float): Decrement of synapse permanence for inactive connections.
    """

    syn_perm_active_inc = 0.1
    syn_perm_inactive_dec = 0.03

    def __init__(self, neuron_layer, num_columns=500, connected_perm=0.5, initial_perm_range=0.2, sparsity=0.1, boost_strength=1):
        """
        Initializes the FFProximalConnection layer.

        Parameters:
            neuron_layer (NeuronLayer): Neuron layer object to connect to.
            num_columns (int): Number of columns in the HTM layer.
            connected_perm (float): Threshold for a synapse to be considered connected.
            initial_perm_range (float): Range for initializing the synapse permanences.
            sparsity (float): Target sparsity of the active columns.
            boost_strength (float): Strength of boosting for underutilized columns.
        """
        self.neuron_layer = neuron_layer
        self.input_size = neuron_layer.size
        self.num_columns = num_columns
        self.connected_perm = connected_perm
        self.initial_perm_range = initial_perm_range
        self.sparsity = sparsity
        self.stimulus_threshold = 5
        self.active_duty_cycle = np.zeros(self.num_columns)
        self.overlap_duty_cycle = np.zeros(self.num_columns)
        self.boosting_values = np.ones(self.num_columns)
        self.boost_strength = boost_strength
        self.step = 1
        self.permanences = self._initialize_permanences()
        self.connected_synapses = self.permanences >= self.connected_perm

    def _initialize_permanences(self):
        """
        Initializes the synapse permanences randomly within a specified range.

        Returns:
            numpy.ndarray: Array of synapse permanences.
        """
        perms = np.random.uniform(self.connected_perm - self.initial_perm_range, self.connected_perm + self.initial_perm_range, (self.num_columns, self.input_size))
        return perms

    def compute_overlap(self, active_indices):
        """
        Computes the overlap of the active neurons with each column.

        Parameters:
            active_indices (list): Indices of active neurons in the neuron layer.

        Returns:
            numpy.ndarray: Array of overlap scores for each column.
        """
        input_vector = np.zeros(self.input_size)
        input_vector[active_indices] = 1
        overlaps = np.dot(self.connected_synapses.astype(float), input_vector)
        overlaps *= self.boosting_values
        return overlaps

    def get_active_columns(self, overlaps):
        """
        Determines the active columns based on the overlap values and sparsity constraint.

        Parameters:
            overlaps (numpy.ndarray): Overlap scores for each column.

        Returns:
            list: Indices of the active columns.
        """
        active_columns = []
        k = int(self.num_columns * self.sparsity)  # Desired number of active columns

        # Sort overlaps and find the kth largest value
        sorted_overlaps = np.sort(overlaps)[::-1]
        kth_overlap = sorted_overlaps[max(0, k-1)]

        # Select all columns with overlap greater than the kth_overlap
        return np.where(overlaps >= kth_overlap)[0].tolist()

    def learn(self, active_indices, active_columns):
        """
        Updates the permanences based on the current input vector, focusing only on active columns.

        Parameters:
            active_indices (list): Indices of active neurons in the neuron layer.
            active_columns (list): Indices of active columns.
        """
        input_vector = np.zeros(self.input_size)
        input_vector[active_indices] = 1

        # Convert active columns list to a boolean mask
        active_column_mask = np.zeros(self.num_columns, dtype=bool)
        active_column_mask[active_columns] = True

        # Create a mask for active inputs
        active_input_mask = input_vector.astype(bool)

        # Decrement all synapses of active columns
        self.permanences[active_column_mask, :] -= FFProximalConnection.syn_perm_inactive_dec

        # Increment synapses that connect active columns to active inputs
        increment_mask = np.outer(active_column_mask, active_input_mask)
        self.permanences += increment_mask * (FFProximalConnection.syn_perm_active_inc + FFProximalConnection.syn_perm_inactive_dec)

        # Ensuring permanences stay within bounds
        self.permanences = np.clip(self.permanences, 0, 1)

    def update_active_duty_cycle(self, col_idx, isactive):
        """
        Updates the active duty cycle for a given column.

        Parameters:
            col_idx (int): The index of the column.
            isactive (bool): Whether the column is active.
        """
        num_timesteps = min(500, self.step)
        self.active_duty_cycle[col_idx] = (self.active_duty_cycle[col_idx] * (num_timesteps - 1) + isactive) / num_timesteps

    def update_overlap_duty_cycle_and_adjust_perm(self, col_idx, current_overlap):
        """
        Updates the overlap duty cycle for a column and adjusts its permanences if necessary.

        Parameters:
            col_idx (int): The index of the column.
            current_overlap (int): The current overlap value for the column.
        """
        min_duty_cycle_fraction = 0.01
        num_timesteps = min(500, self.step)
        self.overlap_duty_cycle[col_idx] = (self.overlap_duty_cycle[col_idx] * (num_timesteps - 1) + (current_overlap > self.stimulus_threshold)) / num_timesteps
        max_duty_cycle = np.max(self.overlap_duty_cycle)
        min_duty_cycle = min_duty_cycle_fraction * max_duty_cycle
        if self.overlap_duty_cycle[col_idx] < min_duty_cycle:
            increase_amount = 0.1 * self.connected_perm
            self.permanences[col_idx] += increase_amount
            self.permanences[col_idx] = np.clip(self.permanences[col_idx], 0, 1)

    def boost_function(self, active_duty_cycle):
        """
        Boosting function for adjusting the active duty cycle.

        Parameters:
            active_duty_cycle (float): The active duty cycle of a column.

        Returns:
            float: The boosted value.
        """
        mean_neighbors_duty_cycle = np.mean(self.active_duty_cycle)
        return np.exp(self.boost_strength * (mean_neighbors_duty_cycle - active_duty_cycle))

    def update_boosting_values(self):
        """
        Updates the boosting values for all columns.
        """
        for c in range(self.num_columns):
            self.boosting_values[c] = self.boost_function(self.active_duty_cycle[c])

    def run_step(self, active_indices):
        """
        Runs a simulation step.

        Parameters:
            active_indices (list): Indices of active neurons in the neuron layer.

        Returns:
            list: Indices of the active columns after the step.
        """
        overlaps = self.compute_overlap(active_indices)
        active_columns = self.get_active_columns(overlaps)
        self.learn(active_indices, active_columns)
        # Update duty cycles and any other stateful properties here
        self.step += 1
        return active_columns