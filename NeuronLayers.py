from FFPConn import FFProximalConnection
from BDDConn import BDDendriticConnection
import numpy as np

class NeuronLayer:
    """
    Represents a generic neuron layer in a Hierarchical Temporal Memory (HTM) system. This is the base class to represent a set of neurons, and will be subclassed for specific neuron layers
    """

    def __init__(self, size):
        """
        Initializes the NeuronLayer.

        Parameters:
            size (int): The number of neurons in this layer.
        """
        self.size = size
        self.active_neurons = []
        self.previous_active_neurons = []

    def update_active_neurons(self, active_indices):
        """
        Updates the list of active neurons.

        Parameters:
            active_indices (list): Indices of currently active neurons.
        """
        self.active_neurons = active_indices

    def get_active_neurons(self):
        """
        Returns the list of active neurons.

        Returns:
            list: Indices of currently active neurons.
        """
        return self.active_neurons
    


class L4Layer(NeuronLayer):
    def __init__(self, size, ffp_input_layer, neurons_per_column):
        """
        Initializes the L4 layer.

        Parameters:
            size (int): The number of columns in the L4 layer.
            ffp_input_layer (NeuronLayer): The input layer for the FFP layer.
            neurons_per_column (int): The number of neurons per column in the L4 layer.
        """
        super().__init__(size)
        self.neurons_per_column = neurons_per_column
        self.ffp_layer = FFProximalConnection(self, ffp_input_layer)
        self.bdd_layers = []
        self.primed_segments_by_layer = {}  # Maps BDD layers to their primed segments

    def add_context_layer(self, bdd_input_layer):
        """
        Adds a BDD layer for context.

        Parameters:
            bdd_input_layer (NeuronLayer): The input layer for the BDD layer.
        """
        bdd_layer = BDDendriticConnection(self, bdd_input_layer)
        self.bdd_layers.append(bdd_layer)

    def run_timestep(self, input_vector):
        """
        Runs a single timestep of processing in the L4 layer.

        Parameters:
            input_vector (list): The sensory input vector for this timestep.
        """
        # Update the FFP layer and determine active columns
        active_columns = self.ffp_layer.run_step(input_vector)

        # Reset the mapping for this timestep
        self.primed_segments_by_layer = {bdd_layer: [] for bdd_layer in self.bdd_layers}

        # Determine predicted neurons and associated segments
        predicted_neurons = set()
        for bdd_layer in self.bdd_layers:
            primed_neurons = bdd_layer.get_primed_neurons()
            predicted_neurons.update(primed_neurons)

            # Store associated segments for each primed neuron
            for neuron_idx in primed_neurons:
                associated_segments = np.where(bdd_layer.segment_to_neuron_map == neuron_idx)[0]
                self.primed_segments_by_layer[bdd_layer].extend(associated_segments)

        # Calculate bursting columns
        bursting_columns = self.calculate_bursting_columns(active_columns, predicted_neurons)

        # Update BDD segments for bursting columns
        for col in bursting_columns:
            for bdd_layer in self.bdd_layers:
                segment_idx = bdd_layer.handle_burst(col)
                if segment_idx == -1:  # Create new segment if needed
                    bdd_layer.create_distal_segment(col)  # Adjust as needed for multiple neurons per column

        # Direct learning for BDD and FFP layers
        self.learn(active_columns, predicted_neurons)

    def calculate_bursting_columns(self, active_columns, predicted_neurons):
        """
        Calculates which columns are bursting.

        Parameters:
            active_columns (list): Active columns from the FFP layer.
            predicted_neurons (set): Predicted neurons from BDD layers.

        Returns:
            list: Columns that are bursting.
        """
        bursting_columns = []
        for col in active_columns:
            # Check if any neuron in the column was predicted
            if not any(neuron in predicted_neurons for neuron in self.get_neurons_in_column(col)):
                bursting_columns.append(col)
        return bursting_columns

    def get_neurons_in_column(self, column_idx):
        """
        Returns the neuron indices for a given column index.

        Parameters:
            column_idx (int): The column index.

        Returns:
            list: Neuron indices in the specified column.
        """
        start_idx = column_idx * self.neurons_per_column
        return list(range(start_idx, start_idx + self.neurons_per_column))

    def learn(self, active_columns):
        # Learning in FFP layer
        self.ffp_layer.learn(active_columns)

        # Learning in BDD layers using stored segment information
        for bdd_layer in self.bdd_layers:
            bdd_layer.learn(self.primed_segments_by_layer[bdd_layer])