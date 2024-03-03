import cupy as cp
from NeuronLayers import NeuronLayer, L2Layer, L4Layer
import random

class Column:
    def __init__(self):
        self.sensory_input = NeuronLayer(self, 150, 1)
        self.location_input = NeuronLayer(self, 2400, 1)
        self.L4 = L4Layer(self, 150, 16, self.sensory_input, 0)
        self.L2 = L2Layer(self, 4096, 1, self.L4, 4, 40, 450)
        self.L4.add_context_connection(self.location_input, 6, True)
        self.L2.add_context_connection(self.L2, 6)
        self.L4.add_context_connection(self.L2, 6, concurrent=False)

    def add_lateral_connection(self, target_column):
        self.L2.add_lateral_connection(target_column, 18)

    def update_activity(self, feature, location):
        self.sensory_input.set_active_neurons(feature)
        self.location_input.set_active_neurons(location)
        self.L4.update_active_neurons(inhibition=0.067)
        self.L2.update_active_neurons()

    def learn(self):
        self.L4.learn()
        self.L2.learn()

    def get_active_counts(self):
        return self.L2.active_neurons.count_nonzero(), self.L4.active_neurons.count_nonzero()