import cupy as cp
from NeuronLayers import NeuronLayer, L2Layer, L4Layer
import random

class Column:
    def __init__(self):
        self.sensory_input = NeuronLayer(self, 150, 1)
        self.location_input = NeuronLayer(self, 2400, 1)
        self.L4 = L4Layer(self, 150, 16, self.sensory_input, 0)
        self.L2 = L2Layer(self, 4096, 1, self.L4, 4, 40, 65)
        self.L4.add_context_connection(self.location_input, 6, True)

    def add_lateral_connection(self, target_column):
        self.L2.add_lateral_connection(target_column, 18)

    def inference_step(self, feature, location):
        self.sensory_input.set_active_neurons(feature)
        self.location_input.set_active_neurons(location)
        self.L4.run_timestep_infer(False, inhibition=0.067)
        self.L2.run_timestep(False, False)

    def training_step(self, feature, location, new_object):
        self.sensory_input.set_active_neurons(feature)
        self.location_input.set_active_neurons(location)
        self.L4.run_timestep_learn(True, inhibition=0.067)
        self.L2.run_timestep(True, new_object)

    def reset_layer(self):
        self.L2.reset_layer()

    def get_active_L2(self):
        return self.L2.active_neurons
        

    