from Column import Column
import random

class MacroColumn:
    def __init__(self, num_minicolumns):
        self.minicolumns = []
        for _ in range(num_minicolumns):
            self.minicolumns.append(Column())
        for col_idx in range(num_minicolumns):
            for target_idx in range(num_minicolumns):
                self.minicolumns[col_idx].add_lateral_connection(self.minicolumns[target_idx])

    def train_on_object(self, input_object):
        new_object = True
        for minicolumn in self.minicolumns:
            minicolumn.reset_layer()
        steps = len(input_object) * 3 - 1
        for i in range(steps):
            for j in range(len(self.minicolumns) - 1):
                feature, location = input_object[(i + j) % len(input_object)]
                self.minicolumns[j].training_step(feature, location, new_object)
            new_object = False
    
    def infer_on_object(self, input_object):
        pass

