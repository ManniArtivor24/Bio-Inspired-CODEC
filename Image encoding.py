import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from tqdm import tqdm
import nest

results_dir = "results/"
rows=174#240
cols=200#320

# Function to convert pixel intensity to current for each color channel
def pixel_intensity_to_current(intensity, offset=380):
    return intensity + offset

# Function to resize image
def resize_image(image, target_size=(cols, rows)):
    if target_size is None:
        return image
    else:
        return cv2.resize(image, target_size)

# Function to simulate raster plot for each image
def simulate_raster_plot(image_file, current_funcs, sim_time1=50.0, pd=13.0):
    # Clear console
    os.system('clear')
    sim_time1=sim_time1+pd
    # Read the image
    print("Reading image {}...".format(image_file))
    image = cv2.imread(os.path.join(images_folder, image_file + image_extension))
    # Resize the image
    resized_image = resize_image(image)
    # Get image dimensions
    height, width, _ = resized_image.shape
    print("Image dimensions: Height: {}, Width: {}".format(height, width))

    # Initialize NEST kernel
    nest.ResetKernel()
    nest.set_verbosity(20)  # Set NEST verbosity level to 20
    nest.SetKernelStatus({'print_time': False})

    # Create layers for Blue, Green, and Red channels
    layers_L1 = []
    layers_L2 = []
    layers_L3 = []
    spikerecorders_L1 = []
    spikerecorders_L2 = []
    spikerecorders_L3 = []
    for i, color in enumerate(['Blue','Green','Red']):
        # Create layer with iaf_psc_alpha neurons
        layers_L1.append(nest.Create('iaf_psc_alpha', width * height))
        layers_L2.append(nest.Create('iaf_psc_alpha', width * height))
        layers_L3.append(nest.Create('iaf_psc_alpha', width * height))
        # Connect each layer to a spike recorder
        spikerecorders_L1.append(nest.Create("spike_recorder"))
        spikerecorders_L2.append(nest.Create("spike_recorder"))
        spikerecorders_L3.append(nest.Create("spike_recorder"))

        # Progress bar for setting currents
        progress_bar = tqdm(total=height * width, desc="Setting currents for {} channel".format(color), position=0,
                            leave=True)

        # Create spike generators for each neuron and inject analog values
        for row in range(height):
            for col in range(width):
                # Calculate the current based on pixel intensity for the corresponding color channel
                intensity = resized_image[row, col, i]
                current = current_funcs[i](intensity)

                # Set current for each neuron
                neuron_index = row * width + col
                nest.SetStatus(layers_L1[i][neuron_index], {"I_e": current})
                nest.Connect(layers_L1[i][neuron_index], layers_L2[i][neuron_index], "one_to_one", syn_spec={"weight": 1200.0})
                nest.Connect(layers_L2[i][neuron_index], layers_L3[i][neuron_index], "one_to_one", syn_spec={"weight": 1200.0})

                # Update progress bar
                progress_bar.update(1)


        nest.Connect(layers_L1[i], spikerecorders_L1[i])
        nest.Connect(layers_L2[i], spikerecorders_L2[i])
        nest.Connect(layers_L3[i], spikerecorders_L3[i])

    # Simulate
    print("Simulating for", image_file)
    nest.Simulate(sim_time1)
    print("Simulation completed for", image_file)

    # Save spike events and senders in HDF5 format
    os.makedirs(results_dir, exist_ok=True)
    with h5py.File(os.path.join(results_dir, image_file + "_spikes.h5"), "w") as file:
        for i, color in enumerate(['Blue', 'Green','Red']):
            events = spikerecorders_L3[i].get("events")
            senders = events["senders"]
            times = events["times"]
            grp = file.create_group(color)
            grp.create_dataset("senders", data=senders)
            grp.create_dataset("times", data=times)
            grp.attrs["image_filename"] = image_file
            grp.attrs["image_dimensions"] = (height, width)
            grp.attrs["simulation_time"] = sim_time1

    # Plot raster plot for each color channel
    plt.figure(figsize=(15, 5))
    for i, color in enumerate(['Blue', 'Green','Red']):
        plt.subplot(1, 3, i + 1)
        plt.title('{} Channel - {} - L3'.format(color, image_file))
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.grid()
        ts = spikerecorders_L3[i].get("events")["times"]
        if(np.min(ts)>pd):
            print("Subtracting the propagation delay:",pd,"ms.")
            ts=ts-pd

        senders = spikerecorders_L3[i].get("events")["senders"]-np.min(spikerecorders_L3[i].get("events")["senders"])  # normalise values between 0 and cols*rows-1
        # print(np.min(senders),np.max(senders),(cols*rows))
        np.save(os.path.join(results_dir, image_file + "_"+ color + "_L3_senders.npy"), senders)
        np.save(os.path.join(results_dir, image_file + "_"+ color + "_L3_ts.npy"), ts)
        plt.vlines(ts, senders, senders + 1, color=color.lower(), linewidths=0.5)
    plt.tight_layout()
    # plt.show()

 # List of image file names
images_folder = 'Frames'  # Adjust this path to point to your Frames folder
image_extension = '.jpg'
target_size = (174, 200)  # Specify the target size for resizing images
results_dir = "results"  # Specify the directory to save
os.makedirs(results_dir, exist_ok=True)

# Get the list of all image files in the folder
image_files = [file[:-len(image_extension)] for file in sorted(os.listdir(images_folder)) if
                   file.endswith(image_extension)]

# Limit the loop to the first 20 frames
for image_file in image_files[:2]:
    simulate_raster_plot(image_file, [pixel_intensity_to_current] * 3)