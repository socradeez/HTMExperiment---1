from FFPConn import FFProximalConnection
from BDDConn import BDDendriticConnection
#from LateralConn import LateralConnection
import cupy as cp
from cupy import sparse


class NeuronLayer:
    def __init__(self, macro_column, num_columns, neurons_per_column):
        self.macro_column = macro_column
        self.num_columns = num_columns
        self.neurons_per_column = neurons_per_column
        self.shape = (num_columns, neurons_per_column)
        self.active_neurons = sparse.csr_matrix(self.shape, dtype=bool)

    def get_active_neurons(self):
        return self.active_neurons
    
    def set_active_neurons(self, array):
        self.active_neurons = array


class L4Layer(NeuronLayer):
    def __init__(self, macro_column, num_columns, neurons_per_column, ffp_input_layer, ffp_stim_thresh):
        super().__init__(macro_column, num_columns, neurons_per_column)
        self.ffp_layer = FFProximalConnection(self, ffp_input_layer, stim_threshold=ffp_stim_thresh)
        self.bdd_layers = []

    def add_context_connection(self, bdd_input_layer, bdd_threshold, concurrent):
        bdd_layer = BDDendriticConnection(self, bdd_input_layer, concurrent=concurrent, activation_threshold=bdd_threshold)
        self.bdd_layers.append(bdd_layer)

    def update_active_neurons(self, inhibition=0.02):
        active_columns = self.ffp_layer.compute_activity(inhibition)
        predicted_indices = cp.empty((0, 2), dtype=int)
        for bdd_layer in self.bdd_layers:
            predicted_cells = bdd_layer.get_predicted_cells()
            predicted_indices = cp.vstack((predicted_indices, predicted_cells))

        predicted_columns_mask = cp.zeros(self.num_columns, dtype=bool)

        if predicted_indices.size > 0:
            predicted_columns = cp.unique(predicted_indices[:, 1])
            predicted_columns_mask[predicted_columns] = True

            extended_active_columns = active_columns[:, None]
            predicted_neurons = cp.zeros((self.num_columns, self.neurons_per_column))
            neuron_indices, col_indices = predicted_indices[:, 0], predicted_indices[:, 1]
            predicted_neurons[col_indices, neuron_indices] = True

            # Set all predicted neurons in active columns to 1
            active_neurons = cp.logical_and(predicted_neurons, extended_active_columns)

        else:
            # If all columns are bursting, set all neurons in active columns
            active_neurons = cp.outer(active_columns, cp.ones(self.neurons_per_column, dtype=bool))

        # Identify bursting columns (columns that are active but not predicted)
        bursting_columns = cp.logical_and(active_columns, cp.logical_not(predicted_columns_mask))
        for col_idx in cp.where(bursting_columns)[0]:
            for bdd_layer in self.bdd_layers:
                bdd_layer.create_distal_segment(col_idx)

        # Set all neurons in bursting columns to 1
        bursting_neurons = cp.outer(bursting_columns, cp.ones(self.neurons_per_column, dtype=bool))
        active_neurons = cp.logical_or(active_neurons, bursting_neurons)
        self.active_neurons = sparse.csr_matrix(active_neurons)

    def learn(self):
        # Learning in FFP layer
        self.ffp_layer.learn()
        self.ffp_layer.step += 1

        # Learning in BDD layers using stored segment information
        for bdd_layer in self.bdd_layers:
            bdd_layer.learn()
