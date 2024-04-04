import numpy as np
import matplotlib.pyplot as plt

# Function to chunk data into chunks of specified duration
def chunk_data_info(data, rate, duration_sec):
    chunk_size = int(rate * duration_sec)
    num_chunks = len(data) // chunk_size
    return num_chunks, chunk_size

def create_normalised_3D_matrix(data_name, times_left, senders_left, times_right, senders_right, rows, cols):
    normalised_3Dmat = np.zeros((rows, cols, 2))
    print("Starting normalisation ...\nProcessing left...")
    for i in range(len(times_left)):
        normalised_3Dmat[int(senders_left[i]), int(np.round(times_left[i], 0)), 0] = 1.0  # left
    print("Left completed ...\nProcessing right...")
    for i in range(len(times_right)):
        normalised_3Dmat[int(senders_right[i] - 1), int(np.round(times_right[i], 0)), 1] = 1.0  # right
    np.save(f"sound_results/normalised_3Dmat_{data_name}.npy", normalised_3Dmat, allow_pickle=True)
    print("Normalisation completed!")
    return normalised_3Dmat.astype(bool)


def raster_plot(normalised_3D_matrix, data):
    """
    Plot raster plot for spike events split between left and right channels.

    Parameters:
        normalised_3D_matrix (numpy.ndarray): 3D matrix split between left and right channels.
        data_num (int): Identifier for the data set.

    Returns:
        None
    """
    print("Preparing left raster plot...")
    num_neurons = normalised_3D_matrix.shape[0]

    # Plot raster plot for left channel
    plt.figure(figsize=(10, 6))
    for neuron_idx in range(num_neurons):
        left_spikes = np.where(normalised_3D_matrix[neuron_idx, :, 0])[0]
        plt.vlines(left_spikes, neuron_idx, neuron_idx + 1, color='red',
                   label='Left Channel' if neuron_idx == 0 else None)
    print("Preparing right raster plot...")
    # Plot raster plot for right channel
    for neuron_idx in range(num_neurons):
        right_spikes = np.where(normalised_3D_matrix[neuron_idx, :, 1])[0]
        plt.vlines(right_spikes, neuron_idx, neuron_idx + 1, color='blue',
                   label='Right Channel' if neuron_idx == 0 else None)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title(f'Raster Plot for {data}')
    plt.legend()
    plt.grid()
    plt.savefig(f"sound_results/{data}_raster_plot.png")
    plt.show()


# Load the data from npy files
data1 = np.load('sound_results/data1.npy')
rate1 = np.load('sound_results/rate1.npy')
senders_left = []
ts_left = []
senders_right = []
ts_right = []
for data_num in ["1"]:
    senders_left.append(np.load(f'sound_results/data{data_num}_left_senders.npy'))
    ts_left.append(np.load(f'sound_results/data{data_num}_left_ts.npy'))
    senders_right.append(np.load(f'sound_results/data{data_num}_right_senders.npy'))
    ts_right.append(np.load(f'sound_results/data{data_num}_right_ts.npy'))

data_shape = [data1.shape]

reconstructed_data1 = np.zeros(data_shape[0])

num_neurons_per_sample = 6
samp_period = 1 / 30
time_step = 60

data1_num_chunks, data1_chunk_size = chunk_data_info(data1, rate1, samp_period)
num_total_neurons = [data1_chunk_size * num_neurons_per_sample]
senders_left[0] = senders_left[0] - 1  # normalisation
senders_right[0] = senders_right[0] - num_total_neurons[0]  # normalisation
print(np.min(senders_left[0]), np.max(senders_left[0]))
print(np.min(senders_right[0]), np.max(senders_right[0]))

datas = ["data1"]

# print(data1_num_total_neurons)
for i in range(len(datas)):
    print(f"++++++++++++++++++ Processing {datas[i]}... +++++++++++++++++++++++")
    num_cols = int(np.max([np.max(ts_left[i]), np.max(ts_right[i])])) + 1
    normalised_3Dmat = create_normalised_3D_matrix(datas[i], ts_left[i], senders_left[i], ts_right[i], senders_right[i],
                                                   num_total_neurons[i], num_cols)
    raster_plot(normalised_3Dmat, datas[i])