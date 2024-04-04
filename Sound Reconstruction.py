import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_idx_for_spike_count(spike_count):
    """
    This function takes a spike count and returns the index if it exists in the current_spikes_values numpy array.
    If the spike count does not exist in the array, it returns -1.

    Parameters:
        spike_count (int): The spike count to search for.

    Returns:
        int: The index if the spike count exists, otherwise -1.
    """
    # Load the current_spikes_values from npy file
    spike_values = np.load('sound_results/current_spikes_values.npy')[:, 1]

    # Iterate through the current_spikes_values array to find the index corresponding to the spike count
    for i in range(0, len(spike_values)):
        if spike_values[i] == spike_count:
            return i
    return -1

    # If the spike count does not exist in the array, return -1
    return -1


def reconstruct_value_from_values(values):
    """
    Reconstructs the value from the given vector of values.

    Parameters:
        currents (numpy.ndarray): Array containing currents for each neuron.

    Returns:
        int: Reconstructed value.
    """

    # Extract the indices of the spike counts corresponding to the given currents
    idx_neuron0 = get_idx_for_spike_count(values[0])
    idx_neuron1 = get_idx_for_spike_count(values[1])
    idx_neuron2 = get_idx_for_spike_count(values[2])
    idx_neuron3 = get_idx_for_spike_count(values[3])
    idx_neuron4 = get_idx_for_spike_count(values[4])
    idx_neuron5 = get_idx_for_spike_count(values[5])

    # Reconstruct the value based on the indices
    value = idx_neuron0 + \
            idx_neuron1 * 10 + \
            idx_neuron2 * 100 + \
            idx_neuron3 * 1000 + \
            idx_neuron4 * 10000

    # Check if the value should be negative
    if idx_neuron5 == 1:
        value *= -1
    return value


def calculate_spike_counts(neuron_matrix):
    """
    Calculate the spike counts for each neuron in a matrix.

    Parameters:
        neuron_matrix (numpy.ndarray): Matrix representing spike events of 6 neurons over 60 ms.

    Returns:
        numpy.ndarray: A vector containing the spike counts for each neuron.
    """
    # Sum along the columns to count spikes for each neuron
    spike_counts = np.sum(neuron_matrix, axis=1)
    return spike_counts


def reconstruct_value(normalised_3Dmat, x, y, data):
    """
    Reconstructs the value from a chunk of 6 neurons and 60 ms of simulation.

    Parameters:
        normalised_3Dmat (numpy.ndarray): 3D matrix containing spike events split between left and right channels.
        x (int): Number of chunks along the time axis.
        y (int): Number of chunks along the neuron axis.

    Returns:
        numpy.ndarray: Reconstructed values for each chunk.
    """
    simulation_ts = 60  # ms
    number_neurons_sample = 6
    reconstructed_vector = np.zeros((x, y, 2))

    # Extract the chunk of spike events for the given chunk index
    idx = 0
    for ts in tqdm(range(0, normalised_3Dmat.shape[1], simulation_ts), desc="Processing simualtion time-steps"):
        for neuron in range(0, normalised_3Dmat.shape[0], number_neurons_sample):
            chunk_left = normalised_3Dmat[neuron:neuron + number_neurons_sample, ts:ts + simulation_ts, 0]
            spike_counts_left = np.sum(chunk_left, axis=1)
            reconstructed_value_left = reconstruct_value_from_values(spike_counts_left)

            chunk_right = normalised_3Dmat[neuron:neuron + number_neurons_sample, ts:ts + simulation_ts, 1]
            spike_counts_right = np.sum(chunk_right, axis=1)
            reconstructed_value_right = reconstruct_value_from_values(spike_counts_right)

            reconstructed_vector[idx, 0] = reconstructed_value_left
            reconstructed_vector[idx, 1] = reconstructed_value_right
            idx += 1
    reconstructed_vector.dump(f"sound_results/reconstructed_{data}.npy")
    return reconstructed_vector


def plot_reconstructed_vector(reconstructed_data, original_data, data_name):
    """
    Plot the reconstructed vector.

    Parameters:
        reconstructed_vector (numpy.ndarray): Reconstructed values for each chunk.
        data_num (int): Identifier for the data set.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(original_data[:, 0], label='Left Channel', color='red')
    plt.plot(original_data[:, 1], label='Right Channel', color='blue')
    plt.xlabel('Samples (divide by 44.8k to get time in ms')
    plt.ylabel('Original amplitude')
    plt.title(f'Original {data_name}')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(reconstructed_data[:, 0], label='Left Channel', color='red')
    plt.plot(reconstructed_data[:, 1], label='Right Channel', color='blue')
    plt.xlabel('Samples (divide by 44.8k to get time in ms')
    plt.ylabel('Amplitude')
    plt.title(f'Reconstructed {data_name}')
    plt.legend()
    plt.grid()
    plt.savefig(f"sound_results/{data_name}_results.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(reconstructed_data[:, 0], label='Left Channel', color='red')
    plt.plot(reconstructed_data[:, 1], label='Right Channel', color='blue')
    plt.xlabel('Samples (divide by 44.8k to get time in ms')
    plt.ylabel('Amplitude')
    plt.title(f'Reconstructed {data_name} signal')
    plt.legend()
    plt.grid()
    plt.savefig(f"sound_results/reconstructed_{data_name}.png")
    plt.show()


data_names = ["audio"]
original_signals = [np.load('sound_results/data1.npy')]
original_rates = [np.load('sound_results/rate1.npy')]
normalised_3Dmats = [np.load("sound_results/normalised_3Dmat_data1.npy", allow_pickle=True)]

for i in range(len(data_names)):
    reconstructed_matrix = reconstruct_value(normalised_3Dmats[i], original_signals[i].shape[0],
                                             original_signals[i].shape[1], data_names[i])
    plot_reconstructed_vector(reconstructed_matrix, original_signals[i], data_names[i])