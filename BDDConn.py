import cupy as cp
from cupy import sparse
import random

class BDDendriticConnection:
    """
    Represents Basal Distal Dendritic Connections in a Hierarchical Temporal Memory (HTM) system.
    """
    syn_perm_active_inc = 0.1
    syn_perm_inactive_dec = 0.03
    syn_creation_prob = 0.5  # Probability of creating a new synapse

    def __init__(self, parent_layer, input_layer, concurrent=False, activation_threshold=5, learning_threshold=5, connected_perm=0.5, initial_perm_range=0.2):

        self.parent_layer = parent_layer
        self.input_layer = input_layer
        self.connected_perm = connected_perm
        self.initial_perm_range = initial_perm_range
        self.activation_threshold = activation_threshold
        self.learning_threshold = learning_threshold
        self.permanences = cp.empty((0, self.input_layer.num_columns, self.input_layer.neurons_per_column), dtype=cp.float64)  # No segments initially
        self.segment_to_neuron_map = cp.empty((0, 2), dtype=int)  # Each row is a pair (row_idx, col_idx)
        self.segments_per_parent_neuron = cp.zeros((self.parent_layer.num_columns, self.parent_layer.neurons_per_column))
        self._concurrent = concurrent

    @property
    def input_array(self):
        if self._concurrent:
            return self.input_layer.active_neurons
        else:
            return self.input_layer.previous_active_neurons
    
    @property
    def active_segments(self):
        connected_synapse_mask = (self.permanences >= self.connected_perm).astype(int)
        active_connected_synapses = cp.tensordot(connected_synapse_mask, self.input_array.A, axes=([1, 2], [0, 1]))
        active_segments_mask = active_connected_synapses > self.activation_threshold
        return active_segments_mask
    
    @property
    def matching_segments(self):  
        potential_synapse_mask = (self.permanences > 0).astype(int)
        active_matching_segments = cp.tensordot(potential_synapse_mask, self.input_array.A, axes=([1, 2], [0, 1]))
        matching_segments_mask = active_matching_segments > self.learning_threshold
        return matching_segments_mask

    def get_predicted_cells(self):
        # Check if there are any active segments
        active_segments = self.active_segments
        if cp.any(active_segments):
            predicted_cells = self.segment_to_neuron_map[active_segments]
        else:
            # If no active segments, return an empty array
            predicted_cells = cp.empty((0, self.segment_to_neuron_map.shape[1]))
        return predicted_cells
    
    def get_active_segments_by_cell(self):
        active_segments_mask = self.active_segments
        # Extract active segment indices
        active_segment_indices = active_segments_mask.nonzero()[0]
        active_segments = self.segment_to_neuron_map[active_segment_indices]

        if active_segments.size == 0:
            return cp.zeros((self.parent_layer.num_columns, self.parent_layer.neurons_per_column), dtype=int)

        # Calculate the total number of columns in the parent layer
        total_columns = self.parent_layer.num_columns

        # Convert 2D indices (row, col) into a unique 1D index
        unique_indices = active_segments[:, 0] * total_columns + active_segments[:, 1]

        # Count the occurrences of each unique index
        counts = cp.bincount(unique_indices, minlength=self.parent_layer.num_columns * self.parent_layer.neurons_per_column)

        # Reshape the counts to the shape of parent_layer.active_neurons
        active_segments_count = counts.reshape(self.parent_layer.num_columns, self.parent_layer.neurons_per_column)

        return active_segments_count
    
    def create_distal_segments(self, requested_columns):
        requested_columns = requested_columns.reshape(self.parent_layer.num_columns)
        # Get the number of requested colums
        num_new_segments = int(cp.sum(requested_columns))

        # Get the index of each requested column
        selected_columns_indices = cp.where(requested_columns)[0]

        # Get the count of segments for each neuron in the selected columns
        num_segments_per_neuron = self.segments_per_parent_neuron[selected_columns_indices]
        
        # Get the indices of the neuron with the fewest segments in each selected column
        indices = cp.argmin(num_segments_per_neuron, axis=1)

        # Determine initial permanences for the new segment
        initial_permanences = cp.random.uniform(self.connected_perm - self.initial_perm_range, 
                                                self.connected_perm + self.initial_perm_range, 
                                                (num_new_segments, self.input_layer.num_columns, self.input_layer.neurons_per_column))
        
        # Set permanences ONLY for active neurons in the input layer
        activity_mask = cp.repeat(self.input_array.A[cp.newaxis, :, :], initial_permanences.shape[0], axis=0)
        new_segments = initial_permanences * activity_mask

        # Use the column and row indices 
        neuron_index_pairs = cp.stack((indices, selected_columns_indices), axis=1)

        # Add the new segment to the permanence matrix and update the mapping array
        if self.permanences.size == 0:
            self.permanences = new_segments
            self.segment_to_neuron_map = neuron_index_pairs
        else:
            self.permanences = cp.concatenate((self.permanences, new_segments), axis=0)
            self.segment_to_neuron_map = cp.concatenate((self.segment_to_neuron_map, neuron_index_pairs), axis=0)

    def learn(self):
        # Reshape and broadcast active_segments_mask to match the shape of perms
        active_segments_mask_broadcasted = self.active_segments[:, cp.newaxis, cp.newaxis]
        active_segments_mask_broadcasted = cp.broadcast_to(active_segments_mask_broadcasted, self.permanences.shape)

        # Create active input mask
        active_input_mask = self.input_layer.active_neurons.A > 0

        # Increment for Active Segments with Active Inputs
        increment_mask = active_segments_mask_broadcasted & active_input_mask
        self.permanences[increment_mask] += self.syn_perm_active_inc

        # Decrement for Active Segments with Inactive Inputs
        inactive_input_mask = ~active_input_mask
        decrement_mask_active = active_segments_mask_broadcasted & inactive_input_mask
        self.permanences[decrement_mask_active] -= self.syn_perm_inactive_dec

        matching_segments_mask_broadcasted = self.matching_segments[:, cp.newaxis, cp.newaxis]
        matching_segments_mask_broadcasted = cp.broadcast_to(matching_segments_mask_broadcasted, self.permanences.shape)
        # Decrement for Matching Inactive Segments with Active Inputs
        decrement_mask_matching = matching_segments_mask_broadcasted & ~active_segments_mask_broadcasted & active_input_mask
        self.permanences[decrement_mask_matching] -= self.syn_perm_inactive_dec

        # Ensure permanences stay within bounds
        self.permanences = cp.clip(self.permanences, 0, 1)


        
