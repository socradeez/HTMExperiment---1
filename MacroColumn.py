from Column import Column
import random

class MacroColumn:
    def __init__(self, num_minicolumns):
        self.minicolumns = []
        for _ in range(num_minicolumns):
            self.minicolumns.append(Column())
        for col_idx in range(num_minicolumns):
            for target_idx in range(num_minicolumns):
                if col_idx != target_idx:
                    self.minicolumns[col_idx].add_lateral_connection(self.minicolumns[target_idx])
    
    def sense_object(self, object, learn):
        for minicolumn in self.minicolumns:
            index = random.randint(0, len(object)-1)
            feature, location = object.pop(index)
            minicolumn.sense_feature(feature, location, learn)