from FFPConn import FFProximalConnection
from BDDConn import BDDendriticConnection
from LateralConn import LateralConnection
import cupy as cp
from cupy import sparse


class NeuronLayer:
    def __init__(self, macro_column, num_columns, neurons_per_column):
        self.macro_column = macro_column
        self.num_columns = num_columns
        self.neurons_per_column = neurons_per_column
        self.shape = (num_columns, neurons_per_column)
        self.active_neurons = sparse.csr_matrix(self.shape, dtype=bool)
        self.previous_active_neurons = sparse.csr_matrix(self.shape, dtype=bool)

    def get_active_neurons(self):
        return self.active_neurons
    
    def set_active_neurons(self, array):
        self.active_neurons = array


class L4Layer(NeuronLayer):
    def __init__(self, macro_column, num_columns, neurons_per_column, ffp_input_layer, ffp_stim_thresh):
        super().__init__(macro_column, num_columns, neurons_per_column)
        self.ffp_layer = FFProximalConnection(self, ffp_input_layer, stim_threshold=ffp_stim_thresh)
        self.bdd_layers = []

    @property
    def predicted_indices(self):
        predicted_indices = cp.empty((0, 2), dtype=int)
        for bdd_layer in self.bdd_layers:
            predicted_cells = bdd_layer.get_predicted_cells()
            predicted_indices = cp.vstack((predicted_indices, predicted_cells))
        return predicted_indices

    def add_context_connection(self, bdd_input_layer, bdd_threshold, concurrent):
        bdd_layer = BDDendriticConnection(self, bdd_input_layer, concurrent=concurrent, activation_threshold=bdd_threshold)
        self.bdd_layers.append(bdd_layer)

    def update_active_neurons(self, inhibition=0.02):
        active_columns = self.ffp_layer.compute_activity(inhibition)
        predicted_indices = self.predicted_indices

        predicted_columns_mask = cp.zeros(self.num_columns, dtype=bool)

        if predicted_indices.size > 0:
            predicted_indices = predicted_indices.astype(int)
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
        if cp.any(bursting_columns):
            for connection in self.bdd_layers:
                connection.create_distal_segments(bursting_columns)

        # Set all neurons in bursting columns to 1
        bursting_neurons = cp.outer(bursting_columns, cp.ones(self.neurons_per_column, dtype=bool))
        active_neurons = cp.logical_or(active_neurons, bursting_neurons)
        self.previous_active_neurons = self.active_neurons
        self.active_neurons = sparse.csr_matrix(active_neurons)

    def learn(self):
        # Learning in FFP layer
        self.ffp_layer.learn()
        self.ffp_layer.step += 1

        # Learning in BDD layers using stored segment information
        for bdd_layer in self.bdd_layers:
            bdd_layer.learn()

class L2Layer(NeuronLayer):

    def __init__(self, macro_column, num_columns, neurons_per_column, ffp_input_layer, ffp_stim_thresh, min_active_neurons, max_active_neurons):
        super().__init__(macro_column, num_columns, neurons_per_column)
        self.ffp_layer = FFProximalConnection(self, ffp_input_layer, stim_threshold=ffp_stim_thresh)
        self.bdd_layers = []
        self.lc_layers = []
        self.min_active_neurons = min_active_neurons
        self.max_active_neurons = max_active_neurons

    def add_context_connection(self, bdd_input_layer, bdd_threshold):
        bdd_layer = BDDendriticConnection(self, bdd_input_layer, concurrent=False, activation_threshold=bdd_threshold)
        self.bdd_layers.append(bdd_layer)

    def add_lateral_connection(self, target_column, lc_threshold):
        lc_layer = LateralConnection(self, target_column.L2, concurrent=False, activation_threshold=lc_threshold)
        self.lc_layers.append(lc_layer)

    def update_active_neurons(self, inhibition=1):
        # Get feed forward activity
        self.ff_activity = self.ffp_layer.compute_activity(inhibition).reshape((self.num_columns, self.neurons_per_column))

        # Get the number of active predictive segments per cell by looping over LC and BDD connections
        active_segments_by_neuron = cp.zeros((self.num_columns, self.neurons_per_column), dtype=int)
        for connection in self.lc_layers:
            active_segments_by_neuron += connection.get_active_segments_by_cell()
        for connection in self.bdd_layers:
            active_segments_by_neuron += connection.get_active_segments_by_cell()

        supported_neurons_mask = cp.clip(active_segments_by_neuron, 0, 1)

        if cp.sum(supported_neurons_mask) < self.min_active_neurons:
            active_neurons = self.ff_activity

            if cp.sum(active_neurons) > self.max_active_neurons:
                sparsity = self.max_active_neurons / (self.num_columns * self.neurons_per_column)
                sparsity_mask = cp.random.rand(self.num_columns, self.neurons_per_column) < sparsity
                active_neurons = active_neurons * sparsity_mask

            segment_candidates = active_neurons - supported_neurons_mask
            segment_candidates = cp.clip(segment_candidates, 0, 1)
            for lclayer in self.lc_layers:
                lclayer.create_distal_segments(segment_candidates)
            for bdlayer in self.bdd_layers:
                bdlayer.create_distal_segments(segment_candidates)
            
        else:
            support_threshold = cp.sort(active_segments_by_neuron.ravel())[-self.min_active_neurons]
            supported_neurons = active_segments_by_neuron >= support_threshold
            active_neurons = cp.logical_and(self.ff_activity, supported_neurons)
            segment_candidates = active_neurons - supported_neurons_mask
            segment_candidates = cp.clip(segment_candidates, 0, 1)
            print('segment candidates are = ', cp.sum(segment_candidates))
            for lclayer in self.lc_layers:
                lclayer.create_distal_segments(segment_candidates)
            for bdlayer in self.bdd_layers:
                bdlayer.create_distal_segments(segment_candidates)

        self.previous_active_neurons = self.active_neurons
        self.active_neurons = sparse.csr_matrix(active_neurons)

    def learn(self):
        # Learning in FFP layer
        self.ffp_layer.learn()
        self.ffp_layer.step += 1

        # Learning in BDD and LC layers
        for bdd_layer in self.bdd_layers:
            bdd_layer.learn()
        for lc_layer in self.lc_layers:
            lc_layer.learn()

    def reset_layer(self):
        self.previous_active_neurons = sparse.csr_matrix(self.shape, dtype=bool)
