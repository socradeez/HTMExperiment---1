import cProfile
from MacroColumn import MacroColumn
from data_generation import generate_objects
import h5py

objectlist = generate_objects(10, 10, 30, 30)


agent = MacroColumn(3)
def main():
    with h5py.File('training_data.h5', 'w') as f:
        for index, input_object in enumerate(objectlist):
            agent.train_on_object(input_object)
            group = f.create_group(f'object_{index}')
            # Convert the sparse matrix to dense if necessary
            active_neurons = agent.minicolumns[0].get_active_L2().A.get()
            # Store the dense matrix in HDF5
            group.create_dataset(f'minicolumn', data=active_neurons)

# Run the profiler on the main function
main()
