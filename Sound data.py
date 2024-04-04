import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

sound_dir = 'sound_results'
os.makedirs(sound_dir, exist_ok=True)
# Function to display sound file information
def display_info(data, rate):
    num_samples = len(data)
    length = len(data) / rate
    min_val = np.min(data)
    max_val = np.max(data)
    print("Number of samples:", num_samples)
    print("Length (seconds):", length)
    print("Minimum value:", min_val)
    print("Maximum value:", max_val)

# Read the WAV files
rate1, data1 = wavfile.read('/home/ntu-user/PycharmProjects/Assesment/Sound/extracted_audio.wav')

# Display information about each sound file
print("Sound Information:")
display_info(data1, rate1)

# Plot the data
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(data1, color='blue')  # Plot data1 in blue
plt.title('Sound')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Store the data in npy files
np.save('sound_results/data1.npy', data1)
np.save('sound_results/rate1.npy', rate1)
