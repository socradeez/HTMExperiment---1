from Column import Column
import random
import h5py

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
        steps = len(input_object) * 3
        for i in range(steps):
            print('i = ', i)
            for j in range(len(self.minicolumns)):
                print('j = ', j)
                feature, location = input_object[(i + j) % len(input_object)]
                self.minicolumns[j].training_step(feature, location, new_object)
            new_object = False

        '''with h5py.File('training_data.h5', 'w') as f:
            for step in range(steps):
                group = f.create_group(f'step_{step}')
                for j, minicolumn in enumerate(self.minicolumns):
                    # Convert the sparse matrix to dense if necessary
                    active_neurons = minicolumn.get_active_L2().A.get()
                    # Store the dense matrix in HDF5
                    group.create_dataset(f'minicolumn_{j}', data=active_neurons)'''
    
    def infer_on_object(self, input_object):
        pass

