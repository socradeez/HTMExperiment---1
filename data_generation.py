import cupy as cp
import cupyx.scipy.sparse as sparse
import numpy as np

def generate_locations(num_arrays):
    array_size = 2400
    active_bits = 10
    sparse_arrays = []

    for _ in range(num_arrays):
        dense_array = cp.zeros((array_size, 1), dtype=int)
        indices = cp.random.choice(array_size, active_bits, replace=False)
        dense_array[indices] = 1
        sparse_array = sparse.csr_matrix(dense_array.astype(bool))
        sparse_arrays.append(sparse_array)

    return sparse_arrays

def generate_features(num_arrays):
    array_size = 150
    active_bits = 10
    sparse_arrays = []

    for _ in range(num_arrays):
        dense_array = cp.zeros((array_size, 1), dtype=int)
        indices = cp.random.choice(array_size, active_bits, replace=False)
        dense_array[indices] = 1
        sparse_array = sparse.csr_matrix(dense_array.astype(bool))
        sparse_arrays.append(sparse_array)

    return sparse_arrays


def generate_objects(num_objects, num_features, feature_library_size, location_library_size):
    '''
    This function generates an array of size (num_objects) which contains individual arrays
    consisting of a feature + location pair taken randomly from the generated libraries
    '''

    # Generate the libraries
    feature_library = generate_features(feature_library_size)
    location_library = generate_locations(location_library_size)
    objects = []
    for i in range(num_objects):
        # Array to hold the objects
        object = []

        for _ in range(num_features):
            # Randomly select a feature and a location
            feature = feature_library[np.random.randint(0, feature_library_size)]
            location = location_library[np.random.randint(0, location_library_size)]

            # Create a pair of feature and location
            feature_location_pair = (feature, location)

            # Add the pair to the objects list
            object.append(feature_location_pair)
        
        objects.append(object)

    return objects