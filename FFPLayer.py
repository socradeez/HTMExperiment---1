import numpy as np

class FFProximalConnection:
    """
    Represents a FeedForward Proximal Connection layer in a Hierarchical Temporal Memory (HTM) system.

    Attributes:
        syn_perm_active_inc (float): Increment of synapse permanence for active connections.
        syn_perm_inactive_dec (float): Decrement of synapse permanence for inactive connections.
    """

    syn_perm_active_inc = 0.03
    syn_perm_inactive_dec = 0.005

    def __init__(self, input_size, num_columns=500, connected_perm=0.5, initial_perm_range=0.1, sparsity=0.1, boost_strength=1):
        """
        Initializes the FFProximalConnection layer.

        Parameters:
            input_size (int): Size of the input vector.
            num_columns (int): Number of columns in the HTM layer.
            connected_perm (float): Threshold for a synapse to be considered connected.
            initial_perm_range (float): Range for initializing the synapse permanences.
            sparsity (float): Target sparsity of the active columns.
            boost_strength (float): Strength of boosting for underutilized columns.
        """
        self.input_size = input_size
        self.num_columns = num_columns
        self.connected_perm = connected_perm
        self.initial_perm_range = initial_perm_range
        self.sparsity = 100 * sparsity
        self.stimulus_threshold = 2
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

    def compute_overlap(self, input_vector):
        """
        Computes the overlap of the input vector with each column.

        Parameters:
            input_vector (numpy.ndarray): The binary input vector.

        Returns:
            numpy.ndarray: Array of overlap scores for each column.
        """
        overlaps = np.dot(self.connected_synapses.astype(float), input_vector)
        overlaps *= self.boosting_values
        return overlaps

    def get_active_columns(self):
        """
        Determines the active columns based on the overlap values and sparsity constraint.

        Returns:
            list: Indices of the active columns.
        """
        active_columns = []
        k = int(self.num_columns // self.sparsity)
        kth = np.partition(self.overlaps, -k)[-k]
        for column in range(self.num_columns):
            if self.overlaps[column] >= kth:
                active_columns.append(column)
        return active_columns

    def learn(self, input_vector):
        """
        Updates the permanences based on the current input vector.

        Parameters:
            input_vector (numpy.ndarray): The current binary input vector.
        """
        decrement_update_matrix = FFProximalConnection.syn_perm_inactive_dec * (1 - input_vector)
        increment_update_matrix = FFProximalConnection.syn_perm_active_inc * input_vector
        self.permanences -= decrement_update_matrix.reshape(1, -1)
        self.permanences += (increment_update_matrix.reshape(1, -1) * self.connected_synapses)
        self.permanences = np.clip(self.permanences, 0, 1)  # Ensuring permanences stay within bounds

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

    def run_step(self, input_vector):
        """
        Runs a simulation step.

        Parameters:
            input_vector (numpy.ndarray): The current binary input vector.

        Returns:
            list: Indices of the active columns after the step.
        """
        self.overlaps = self.compute_overlap(input_vector)
        self.active_columns = self.get_active_columns()
        self.learn(input_vector)
        for column in range(self.num_columns):
            is_active = column in self.active_columns
            self.update_active_duty_cycle(column, is_active)
            self.update_overlap_duty_cycle_and_adjust_perm(column, self.overlaps[column])
        self.step += 1
        return self.active_columns