import h5py
import numpy as np
import matplotlib.pyplot as plt

# Function to load data from HDF5 file
def load_data_from_hdf5(file_path, step, minicolumn_index):
    with h5py.File(file_path, 'r') as file:
        dataset_name = f'step_{step}/minicolumn_{minicolumn_index}'
        data = np.array(file[dataset_name])
        return data

# Function to update the plot
def update_plot(step, ax, fig, file_path, minicolumn_index, txt):
    active_neurons = load_data_from_hdf5(file_path, step, minicolumn_index)
    print(active_neurons.sum())
    grid = active_neurons.reshape((64, 64))

    # Update the data, title, and text
    ax.imshow(grid, cmap='Greens')
    ax.set_title(f'Step: {step}')
    total_active_neurons = np.sum(active_neurons)
    txt.set_text(f'Total Active Neurons: {total_active_neurons}')

    # Redraw the figure
    fig.canvas.draw()

# Initialize step number and minicolumn index
step_number = 0
minicolumn_index = 0

# File path
file_path = 'training_data.h5'

# Create a figure and axis
fig, ax = plt.subplots()
txt = fig.text(0.7, 0.05, '', fontsize=12)

# Initial plot
update_plot(step_number, ax, fig, file_path, minicolumn_index, txt)

# Event handling function
def on_key(event):
    global step_number
    if event.key == 'right':
        step_number += 1
    elif event.key == 'left' and step_number > 0:
        step_number -= 1
    else:
        return
    update_plot(step_number, ax, fig, file_path, minicolumn_index, txt)

# Connect the event handler
fig.canvas.mpl_connect('key_press_event', on_key)

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Show the plot
plt.show()