import cupy as cp
from NeuronLayers import NeuronLayer, L2Layer, L4Layer
import random

class Column:
    def __init__(self):
        self.sensory_input = NeuronLayer(self, 150, 1)
        self.location_input = NeuronLayer(self, 2400, 1)
        self.L4 = L4Layer(self, 150, 16, self.sensory_input, 0)
        self.L2 = L2Layer(self, 4096, 1, self.L4, 3, 40, 65)
        self.L4.add_context_connection(self.location_input, 6, True)

    def add_lateral_connection(self, target_column):
        self.L2.add_lateral_connection(target_column, 18)

    def sense_feature(self, feature, location, learn=True):
        self.sensory_input.set_active_neurons(feature)
        self.location_input.set_active_neurons(location)
        self.L4.run_timestep(learn, inhibition=0.066)
        self.L2.run_timestep(learn)

    