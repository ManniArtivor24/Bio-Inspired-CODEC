import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nest

rows = 174
cols = 200

directory = 'npy_results'
if not os.path.exists(directory):
    os.makedirs(directory)

def create_normalised_2D_matrix(times, senders, cols):
    rows = np.max(senders) + 1
    normalised_2Dmat = np.zeros((rows, cols))
    for i in range(len(times)):
        normalised_2Dmat[int(senders[i]), int(np.round(times[i], 0))] = 1.0
        normalised_2Dmat.dump(os.path.join(directory, "normalised_2Dmat.npy"))

    # Save the matrix as .npy file
    file_path = os.path.join(directory, "normalised_2Dmat.npy")
    np.save(file_path, normalised_2Dmat)
    return normalised_2Dmat.astype(bool)

def find_spike_pattern(pattern, normalised_2Dmat):
    for idx in range(normalised_2Dmat.shape[0]):
        if np.array_equal(normalised_2Dmat[idx], pattern):
            return idx
    return -1

def spikes_to_analog(pattern, normalised_2Dmat):
    min_distance = np.inf
    best_match_idx = -1
    for idx in range(normalised_2Dmat.shape[0]):
        distance = np.linalg.norm(pattern -normalised_2Dmat[idx])
        if distance < min_distance:
            min_distance = distance
            best_match_idx = idx
    return best_match_idx

#Reset NEST kernel
nest.ResetKernel()

# Define the number of neurons
num_neurons = 256
offset = 380

# Create layer with iaf_psc_alpha neurons
layers_L1 = nest.Create('iaf_psc_alpha', num_neurons)
layers_L2 = nest.Create('iaf_psc_alpha', num_neurons)
layers_L3 = nest.Create('iaf_psc_alpha', num_neurons)

spikerecorders_L1 = nest.Create("spike_recorder")
spikerecorders_L2 = nest.Create("spike_recorder")
spikerecorders_L3 = nest.Create("spike_recorder")

# Define the analog values from 0 to 255
min_analog_value = 0 + offset
max_analog_value = 255 + offset

ind = 0

for value in range(min_analog_value, max_analog_value):
    nest.SetStatus(layers_L1[ind], {"I_e": value})
    nest.SetStatus(layers_L2[ind], {"I_e": value})
    nest.SetStatus(layers_L3[ind], {"I_e": value})
    ind += 1

nest.Connect(layers_L1, spikerecorders_L1)
nest.Connect(layers_L2, spikerecorders_L2)
nest.Connect(layers_L3, spikerecorders_L3)

# Simulate
nest.Simulate(50.0)

# Get spike times
events_L1 = spikerecorders_L1.get("events")
senders_L1 = events_L1["senders"]
ts_L1 = events_L1["times"]

events_L2 = spikerecorders_L2.get("events")
senders_L2 = events_L2["senders"]
ts_L2 = events_L2["times"]

events_L3 = spikerecorders_L3.get("events")
senders_L3 = events_L3["senders"]
ts_L3 = events_L3["times"]

normalised_2Dmat = create_normalised_2D_matrix(ts_L3, senders_L3, 51)

# Load the normalised 2D matrix
print("Loading normalised 2D matrix...")
file_path = os.path.join(directory, "normalised_2Dmat.npy")
if os.path.exists(file_path):
    normalised_2Dmat = np.load(file_path, allow_pickle=True)
    print("Normalised 2D matrix loaded.")
else:
    print(f"Error: The file '{file_path}' does not exist.")
    exit()

# List of figure names
figures = [f'frame{i:04d}' for i in range(1, 11)]

# Create directory for reconstructed images
reconstructed_dir = "reconstructed_images"
os.makedirs(reconstructed_dir, exist_ok=True)

for idx, fig in enumerate(figures):
    # Read events and senders from files
    print(f"Processing {fig} ...")
    file_path = f"results/normalised_3Dmat_{fig}.npy"
    if os.path.exists(file_path):
        normalised_3Dmat = np.load(file_path, allow_pickle=True)
        print(normalised_3Dmat.shape)
    else:
        print(f"Error: The file '{file_path}' does not exist.")
        continue

    # Create 2D matrix for each channel
    analogue_values_ch = np.zeros((174, 200, 3), dtype=np.uint8)
    for k in range(normalised_3Dmat.shape[2]):  # Iterate over the third dimension
        for i in range(normalised_3Dmat.shape[0]):
            tmp = spikes_to_analog(normalised_3Dmat[i, :, k], normalised_2Dmat)
            if tmp < 0:
                raise ValueError("Invalid value!")
            r, c = divmod(i, 200)  # Calculate row and column indices
            analogue_values_ch[r, c, k] = np.array(tmp).astype(np.uint8)

    print("2D matrix created.")

    # Load original image
    print("Loading original image...")
    file_path = f"Frames/{fig}.jpg"
    if os.path.exists(file_path):
        original_image = cv2.imread(file_path)
        original_image_resized = cv2.resize(original_image, (200, 174))
        print("Original image loaded.")
    else:
        print(f"Error: The file '{file_path}' does not exist.")
        continue

    # Calculate the absolute difference between original and reconstructed images
    print("Calculating absolute difference...")
    diff_image = cv2.absdiff(original_image_resized, analogue_values_ch)
    print("Absolute difference calculated.")

    # Save reconstructed image
    reconstructed_image_path = os.path.join(reconstructed_dir, f"{fig}_reconstructed.jpg")
    cv2.imwrite(reconstructed_image_path, cv2.cvtColor(analogue_values_ch, cv2.COLOR_RGB2BGR))
    print(f"Reconstructed image saved at: {reconstructed_image_path}")

print("All processing completed.")
