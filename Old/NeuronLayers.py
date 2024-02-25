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

    def add_context_connection(self, bdd_input_layer, bdd_threshold, concurrent):
        bdd_layer = BDDendriticConnection(self, bdd_input_layer, concurrent=concurrent, activation_threshold=bdd_threshold)
        self.bdd_layers.append(bdd_layer)

    def run_timestep_infer(self, learn, inhibition=0.02):
        self.active_columns = self.ffp_layer.get_activity(inhibition, boosting=True)
        if learn:
            self.ffp_layer.learn()

        predicted_indices = cp.empty((0, 2), dtype=int)
        for bdd_layer in self.bdd_layers:
            predicted_cells = bdd_layer.get_predicted_cells()
            predicted_indices = cp.vstack((predicted_indices, predicted_cells))

        predicted_columns_mask = cp.zeros(self.num_columns, dtype=bool)

        if predicted_indices.size > 0:
            predicted_columns = cp.unique(predicted_indices[:, 1])
            predicted_columns_mask[predicted_columns] = True

            extended_active_columns = self.active_columns[:, None]
            predicted_neurons = cp.zeros((self.num_columns, self.neurons_per_column))
            neuron_indices, col_indices = predicted_indices[:, 0], predicted_indices[:, 1]
            predicted_neurons[col_indices, neuron_indices] = True

            # Set all predicted neurons in active columns to 1
            active_neurons = cp.logical_and(predicted_neurons, extended_active_columns)

        else:
            # If all columns are bursting, set all neurons in active columns
            active_neurons = cp.outer(self.active_columns, cp.ones(self.neurons_per_column, dtype=bool))

        # Identify bursting columns (columns that are active but not predicted)
        bursting_columns = cp.logical_and(self.active_columns, cp.logical_not(predicted_columns_mask))
        for col_idx in cp.where(bursting_columns)[0]:
            for bdd_layer in self.bdd_layers:
                bdd_layer.create_distal_segment(col_idx)

        # Set all neurons in bursting columns to 1
        bursting_neurons = cp.outer(bursting_columns, cp.ones(self.neurons_per_column, dtype=bool))
        active_neurons = cp.logical_or(active_neurons, bursting_neurons)

        self.active_neurons = sparse.csr_matrix(active_neurons)

        # Direct learning for BDD and FFP layers
        if learn:
            self.learn()

        self.previous_active_neurons = self.active_neurons

    def run_timestep_learn(self, learn, inhibition=0.02):
        self.active_columns = self.ffp_layer.get_activity(inhibition, boosting=True)
        if learn:
            self.ffp_layer.learn()

        predicted_indices = cp.empty((0, 2), dtype=int)
        for bdd_layer in self.bdd_layers:
            predicted_cells = bdd_layer.get_predicted_cells()
            predicted_indices = cp.vstack((predicted_indices, predicted_cells))

        predicted_columns_mask = cp.zeros(self.num_columns, dtype=bool)

        if predicted_indices.size > 0:
            predicted_columns = cp.unique(predicted_indices[:, 1])
            predicted_columns_mask[predicted_columns] = True

            extended_active_columns = self.active_columns[:, None]
            predicted_neurons = cp.zeros((self.num_columns, self.neurons_per_column))
            neuron_indices, col_indices = predicted_indices[:, 0], predicted_indices[:, 1]
            predicted_neurons[col_indices, neuron_indices] = True

            # Set all predicted neurons in active columns to 1
            active_neurons = cp.logical_and(predicted_neurons, extended_active_columns)

        else:
            active_neurons = cp.zeros((self.num_columns, self.neurons_per_column))

        # Identify bursting columns (columns that are active but not predicted)
        bursting_columns = cp.logical_and(self.active_columns, cp.logical_not(predicted_columns_mask))
        bursting_winner_neurons = cp.zeros((self.num_columns, self.neurons_per_column), dtype=bool)
        for col_idx in cp.where(bursting_columns)[0]:
            for bdd_layer in self.bdd_layers:
                winner_neuron_row = bdd_layer.create_distal_segment(col_idx)
                bursting_winner_neurons[col_idx, winner_neuron_row] = True

        # Set all neurons in bursting columns to 1
        active_neurons = cp.logical_or(active_neurons, bursting_winner_neurons)

        self.active_neurons = sparse.csr_matrix(active_neurons)

        # Direct learning for BDD and FFP layers
        if learn:
            self.learn()

        self.previous_active_neurons = self.active_neurons


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

    def run_timestep(self, learn, new_object):
        if new_object and learn or (not learn):
            self.ff_activity = self.ffp_layer.get_activity(inhibition=1)
            print('ff overlaps: ', self.ffp_layer.compute_overlaps(True))

            active_count = cp.sum(self.ff_activity)
            if active_count < self.min_active_neurons:
                neurons_to_activate = self.min_active_neurons - active_count

                inactive_indices = cp.where(self.ff_activity.flatten() == 0)

                selected_indices = cp.random.choice(inactive_indices, size=neurons_to_activate, replace=False)

                # Activate the selected neurons
                self.ff_activity.ravel()[selected_indices] = 1

                self.ffp_layer.activity = self.ff_activity
            
            self.ff_activity = self.ff_activity.reshape((self.num_columns, self.neurons_per_column))

            self.active_segments_by_neuron = cp.zeros((self.num_columns, self.neurons_per_column), dtype=int)

            for connection in self.lc_layers:
                self.active_segments_by_neuron += connection.get_active_segments_by_cell()
                print('lclayer shape = ', connection.permanences.shape)

            supported_neurons_mask = cp.clip(self.active_segments_by_neuron, 0, 1)

            if cp.count_nonzero(supported_neurons_mask) < self.min_active_neurons:
                self.active_neurons = self.ff_activity
                neurons_to_activate = self.min_active_neurons - int(cp.count_nonzero(supported_neurons_mask))

                inactive_indices = cp.where(supported_neurons_mask.flatten() == 0)[0].astype(int)
                selected_indices = cp.random.choice(inactive_indices, size=neurons_to_activate, replace=False)

                for index in selected_indices:
                    for layer in self.lc_layers:
                        layer.create_distal_segment(index)
                
            else:
                support_threshold = cp.sort(self.active_segments_by_neuron.ravel())[-self.min_active_neurons]
                supported_neurons = self.active_segments_by_neuron > support_threshold
                self.active_neurons = cp.logical_and(self.ff_activity, supported_neurons)
            self.active_neurons = sparse.csr_matrix(self.active_neurons)

            if self.active_neurons.count_nonzero() < self.max_active_neurons and learn:
                self.learn()
            
            self.previous_active_neurons = self.active_neurons

        else:
            if learn:
                self.learn()

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

    




        
