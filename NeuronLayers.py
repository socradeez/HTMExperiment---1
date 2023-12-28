from FFPConnCP import FFProximalConnection
from BDDConnCP import BDDendriticConnection
from LateralConnCP import LateralConnection
import cupy as cp
from cupy import sparse


class NeuronLayer:
    """
    Represents a generic neuron layer in a Hierarchical Temporal Memory (HTM) system. 
    This is the base class to represent a set of neurons, and will be subclassed for specific neuron layers.
    """

    def __init__(self, macro_column, num_columns, neurons_per_column):
        """
        Initializes the NeuronLayer.

        Parameters:
            size (int): The number of neurons in this layer.
        """
        self.macro_column = macro_column
        self.num_columns = num_columns
        self.neurons_per_column = neurons_per_column
        self.shape = (num_columns, neurons_per_column)
        self.active_neurons = sparse.csr_matrix(self.shape, dtype=bool)
        self.previous_active_neurons = sparse.csr_matrix(self.shape, dtype=bool)

    def get_active_neurons(self):
        """
        Returns the sparse matrix of active neurons.

        Returns:
            cupy.sparse: matrix of neurons with active neurons represented by active bool.
        """
        return self.active_neurons
    
    def set_active_neurons(self, array):
        self.active_neurons = array

class L4Layer(NeuronLayer):
    def __init__(self, macro_column, num_columns, neurons_per_column, ffp_input_layer):
        """
        Initializes the L4 layer. This layer contains minicolumns as groupings of neurons which are fed from a shared proximal dendritic segment. 
        Any number of context layer connections can be added. 

        Parameters:
            size (int): The number of columns in the L4 layer.
            ffp_input_layer (NeuronLayer): The input layer for the FFP layer.
            neurons_per_column (int): The number of neurons per column in the L4 layer.
        """
        super().__init__(macro_column, num_columns, neurons_per_column)
        self.ffp_layer = FFProximalConnection(self, ffp_input_layer, stim_threshold=5)
        self.bdd_layers = []

    def add_context_connection(self, bdd_input_layer, concurrent=False):
        """
        Adds a BDD layer for context.
        This connection type can be temporal (in the context of the previous input), or concurrent (in the context of the current input).
        They can connect to any other neuron layer, though only a temporal delayed connection is available to the parent layer due to recursion.

        Parameters:
            bdd_input_layer (NeuronLayer): The input layer for the BDD layer.
        """
        bdd_layer = BDDendriticConnection(self, bdd_input_layer, concurrent)
        self.bdd_layers.append(bdd_layer)

    def run_timestep(self, inhibition=0.02):
        self.active_columns = self.ffp_layer.get_activity(inhibition, boosting=True)
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

    def __init__(self, macro_column, num_columns, neurons_per_column, ffp_input_layer):

        super().__init__(macro_column, num_columns, neurons_per_column)
        self.ffp_layer = FFProximalConnection(self, ffp_input_layer, stim_threshold=5)
        self.bdd_layers = []
        self.lc_layers = []
        self.lc_threshold = int(0.02 * self.num_columns * self.neurons_per_column)

    def add_context_connection(self, bdd_input_layer, concurrent=False):

        bdd_layer = BDDendriticConnection(self, bdd_input_layer, concurrent)
        self.bdd_layers.append(bdd_layer)

    def add_lateral_connection(self, target_column):

        lc_layer = LateralConnection(self, target_column.layer2, concurrent=False)
        self.lc_layers.append(lc_layer)

    def run_timestep(self):
        self.ff_activity = self.ffp_layer.get_activity(inhibition=1)

        self.active_segments_by_neuron = cp.zeros((self.num_columns, self.neurons_per_column), dtype=int)
        for connection in self.lc_layers:
            self.active_segments_by_neuron += connection.get_active_segments_by_cell()

        # Flatten the array for partitioning
        flat_segments = self.active_segments_by_neuron.ravel()
        # Check if the number of non-zero elements is less than lc_threshold
        if cp.count_nonzero(flat_segments) < self.lc_threshold:
            s = 0
            self.supported_neurons_mask = cp.ones((self.num_columns, self.neurons_per_column))

            # Identify active columns with zero active segments
            columns_with_zero_segments = cp.sum(self.active_segments_by_neuron, axis=1) == 0
            candidate_columns = cp.where(cp.logical_and(self.ff_activity, columns_with_zero_segments))[0]

            # Number of segments to create
            segments_to_create = int(self.lc_threshold - cp.count_nonzero(flat_segments))

            if candidate_columns.size < segments_to_create:
                candidate_columns = cp.arange(self.num_columns)
            # Randomly select columns for new segment creation
            selected_columns = cp.random.choice(candidate_columns, size=segments_to_create, replace=False)

            # Create new segments for the selected columns
            for col_idx in selected_columns:
                for connection in self.lc_layers:
                    connection.create_distal_segment(col_idx)

        else:
            # Find the lc_thresholdth largest element
            threshold_index = len(flat_segments) - self.lc_threshold  # Adjust for largest
            # Negate, partition, and negate again
            
            partitioned = cp.partition(flat_segments, threshold_index)
            s = partitioned[threshold_index]
            self.supported_neurons_mask = self.active_segments_by_neuron >= s

        self.ff_activity = self.ff_activity.reshape(self.num_columns, self.neurons_per_column)

        self.active_neurons = cp.logical_and(self.supported_neurons_mask, self.ff_activity)
        self.active_neurons = sparse.csr_matrix(self.active_neurons)

        if self.active_neurons.count_nonzero() == 0:
            print('ffactivity = ', self.ff_activity)
            print('supported mask = ', self.supported_neurons_mask)

        self.learn()

        self.previous_active_neurons = self.active_neurons

    def learn(self):
        # Learning in FFP layer
        self.ffp_layer.learn()
        self.ffp_layer.step += 1

        # Learning in BDD layers using stored segment information
        for bdd_layer in self.bdd_layers:
            bdd_layer.learn()
        for lc_layer in self.lc_layers:
            lc_layer.learn()

    




        
