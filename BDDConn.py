import numpy as np
import random

class BDDendriticConnection:
    """
    Represents Basal Distal Dendritic Connections in a Hierarchical Temporal Memory (HTM) system.
    """

    syn_perm_active_inc = 0.1
    syn_perm_inactive_dec = 0.03
    syn_creation_prob = 0.5  # Probability of creating a new synapse

    def __init__(self, parent_layer, input_layer, connected_perm=0.5, initial_perm_range=0.2):
        """
        Initializes the BDDendriticConnection layer.

        Parameters:
            parent_layer (NeuronLayer): Neuron layer to which the connections are made.
            input_layer (NeuronLayer): Neuron layer from which the connections originate.
            connected_perm (float): Threshold for a synapse to be considered connected.
            initial_perm_range (float): Range for initializing the synapse permanences.
        """
        self.parent_layer = parent_layer
        self.input_layer = input_layer
        self.connected_perm = connected_perm
        self.initial_perm_range = initial_perm_range
        self.permanences = np.zeros((0, self.input_layer.size))  # No segments initially
        self.segment_to_neuron_map = np.array([], dtype=int)  # Maps segments to neurons

    def create_distal_segment(self, neuron_idx):
        """
        Creates a new distal segment for a given neuron in the parent layer.

        Parameters:
            neuron_idx (int): Index of the neuron in the parent layer for which to create a distal segment.
        """
        new_segment = np.zeros(self.input_layer.size)
        potential_synapses = np.random.rand(self.input_layer.size) < BDDendriticConnection.syn_creation_prob
        active_synapses = potential_synapses & np.isin(range(self.input_layer.size), self.input_layer.previous_active_neurons)
        initial_permanences = np.random.uniform(self.connected_perm - self.initial_perm_range, self.connected_perm, self.input_layer.size)
        new_segment[active_synapses] = initial_permanences[active_synapses]

        # Add the new segment to the permanence matrix and update the mapping array
        self.permanences = np.vstack([self.permanences, new_segment])
        self.segment_to_neuron_map = np.append(self.segment_to_neuron_map, neuron_idx)

    def compute_active_synapses(self):
        """
        Computes active synapses based on the active neurons in the input layer.

        Parameters:
            active_indices (list): Indices of active neurons in the input layer.

        Returns:
            numpy.ndarray: Array indicating active synapses for each segment.
        """
        input_vector = np.zeros(self.input_layer.size)
        input_vector[self.input_layer.active_neurons] = 1
        active_synapses = (np.dot(self.permanences, input_vector) >= self.connected_perm)
        return active_synapses

    def get_primed_neurons(self):
        """
        Determines the primed neurons in the parent layer based on active synapses.

        Returns:
            list: Indices of primed neurons in the parent layer.
        """
        active_synapses = self.compute_active_synapses()
        active_segments = np.where(active_synapses.any(axis=1))[0]
        primed_neurons = self.segment_to_neuron_map[active_segments]
        return np.unique(primed_neurons).tolist()

    def learn(self, parent_layer_neuron_indices):
        """
        Updates the permanences based on the learning rules.

        Parameters:
            parent_layer_neuron_indices (list): Indices of neurons in the parent layer subject to learning.
        """
        for neuron_idx in parent_layer_neuron_indices:
            # Find all segments associated with this neuron
            associated_segments = np.where(self.segment_to_neuron_map == neuron_idx)[0]

            for seg_idx in associated_segments:
                segment_permanences = self.permanences[seg_idx]

                # Increase permanence for synapses connected to previously active neurons
                segment_permanences[self.input_layer.previous_active_neurons] += BDDendriticConnection.syn_perm_active_inc

                # Decrease permanence for synapses not connected to previously active neurons
                inactive_synapses = [i for i in range(self.input_layer.size) if i not in self.input_layer.previous_active_neurons]
                segment_permanences[inactive_synapses] -= BDDendriticConnection.syn_perm_inactive_dec

                # Ensure permanence values stay within bounds
                self.permanences[seg_idx] = np.clip(segment_permanences, 0, 1)

    def handle_burst(self, column_idx):
        """
        Selects a distal segment for a bursting column. If no suitable segment exists, 
        creates a new segment for a randomly selected neuron in the column.

        Parameters:
            column_idx (int): Index of the bursting column in the parent layer.

        Returns:
            int: Index of the selected or newly created segment.
        """
        neurons_per_column = self.parent_layer.neurons_per_column
        start_neuron_idx = column_idx * neurons_per_column
        end_neuron_idx = start_neuron_idx + neurons_per_column

        for neuron_idx in range(start_neuron_idx, end_neuron_idx):
            associated_segments = np.where(self.segment_to_neuron_map == neuron_idx)[0]

            # Check if any of these segments had active synapses
            for seg_idx in associated_segments:
                if np.any(self.permanences[seg_idx] > 0):
                    return seg_idx  # Return the index of the first segment with some activity

        # If no segment is found, create a new segment for a randomly chosen neuron in the column
        random_neuron_idx = random.choice(range(start_neuron_idx, end_neuron_idx))
        self.create_distal_segment(random_neuron_idx)
        new_segment_idx = len(self.permanences) - 1  # Index of the newly created segment
        return new_segment_idx