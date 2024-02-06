import cupy as cp
from cupy import sparse
import random

class BDDendriticConnection:
    """
    Represents Basal Distal Dendritic Connections in a Hierarchical Temporal Memory (HTM) system.
    """
    syn_perm_active_inc = 0.1
    syn_perm_inactive_dec = 0.1
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
        self._concurrent = concurrent

    @property
    def input_array(self):
        if self._concurrent:
            return self.input_layer.previous_active_neurons
        else:
            return self.input_layer.active_neurons
    
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

    def create_distal_segment(self, col_idx):
        # Number of neurons per column
        num_neurons = self.parent_layer.neurons_per_column

        # Count segments for each neuron in the specified column
        segment_counts = cp.zeros(num_neurons, dtype=int)
        for row in range(num_neurons):
            segment_counts[row] = cp.sum((self.segment_to_neuron_map[:, 0] == row) & (self.segment_to_neuron_map[:, 1] == col_idx))

        # Find the neuron(s) with the fewest segments
        min_segments = cp.min(segment_counts)
        neurons_with_min_segments = cp.where(segment_counts == min_segments)[0]

        # Randomly select one of these neurons
        selected_row = cp.random.choice(neurons_with_min_segments, size=1)
        neuron_idx = cp.array([selected_row, cp.array([col_idx])]).reshape(1, 2)

        # Determine initial permanences for the new segment
        initial_permanences = cp.random.uniform(self.connected_perm - self.initial_perm_range, 
                                                self.connected_perm + self.initial_perm_range, 
                                                (self.input_layer.num_columns, self.input_layer.neurons_per_column))
        
        # Set the permanences for active synapses
        new_segment = initial_permanences * self.input_array.A
        new_segment = new_segment.reshape(1, *new_segment.shape)

        # Add the new segment to the permanence matrix and update the mapping array
        self.permanences = cp.vstack((self.permanences, new_segment))
        self.segment_to_neuron_map = cp.vstack((self.segment_to_neuron_map, neuron_idx))

        return(selected_row)

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


        
