import Column
import cupy as cp
from data_generation import generate_objects
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_neuron_changes(prev_dense, current_dense):
    """
    Visualizes changes between two sparse matrix iterations.
    
    Parameters:
    - prev_matrix: The sparse matrix from the previous iteration (numpy array or compatible).
    - current_matrix: The sparse matrix from the current iteration (numpy array or compatible).
    """


    # Define states:
    # 0 -> no activation in both
    # 1 -> activated in current, not in previous (new activation)
    # 2 -> activated in both (consistent activation)
    # 3 -> activated in previous, not in current (deactivation)
    state_matrix = np.zeros(prev_dense.shape, dtype=int) 
    
    test1 = np.zeros(prev_dense.shape, dtype=int)
    test2 = np.zeros(prev_dense.shape, dtype=int)
    test3 = np.zeros(prev_dense.shape, dtype=int)
    # New activations
    state_matrix[(current_dense == 1) & (prev_dense == 0)] = 1
    test1[(current_dense == 1) & (prev_dense == 0)] = 1
    print('new activations = ', cp.sum(test1))
    # Consistent activations
    state_matrix[(current_dense == 1) & (prev_dense == 1)] = 2
    test2[(current_dense == 1) & (prev_dense == 1)] = 1
    print('consistent activations = ', cp.sum(test2))
    # Deactivations
    state_matrix[(current_dense == 0) & (prev_dense == 1)] = 3
    test3[(current_dense == 0) & (prev_dense == 1)] = 1
    print('deactivations = ', cp.sum(test3))

    # Define custom colormap
    cmap = ListedColormap(['white', 'green', 'blue', 'red'])  # white=no activation, green=new, blue=consistent, red=deactivated

    plt.figure(figsize=(10, 6))
    plt.imshow(state_matrix, cmap=cmap, aspect='auto')
    plt.title("Neuron Activation Changes")
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Group")
    #plt.show()

# Example usage (you need to replace 'prev_matrix' and 'current_matrix' with your actual sparse matrices)
# visualize_neuron_changes(prev_matrix, current_matrix)

data = generate_objects(12, 4, 30, 30)

blank_matrix = np.zeros((4096, 1))

testobject = data[0]

column = Column.Column()
column2 = Column.Column()

column.add_lateral_connection(column2)
column2.add_lateral_connection(column)

encoding_dict = {}

for index, featloc in enumerate(testobject):
    encoding_dict[index] = []

for _ in range(10):
    for objindex, testobject in enumerate(data):
        print('new run on object')
        for index, (feature, location) in enumerate(testobject):
            print('starting new feature')
            column.update_activity(feature, location)
            if objindex == 0:
                encoding_dict[index].append(cp.asnumpy(column.L2.active_neurons.A))
            column2.update_activity(feature, location)
            column.learn()
            column2.learn()
            L2count, L4count = column.get_active_counts()
            if objindex == 0 and index == 3:
                print("L4 active neuron count is", L4count)
                print("L2 active neuron count is", L2count)

visualize_neuron_changes(blank_matrix, encoding_dict[3][0])
for x in range(1, 10):
    visualize_neuron_changes(encoding_dict[3][x-1], encoding_dict[3][x])



