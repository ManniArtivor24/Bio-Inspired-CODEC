import cv2
import os

# Directory containing the reconstructed frames
frames_dir = '/home/ntu-user/PycharmProjects/Assesment/reconstructed_images'
output_video_path = 'reconstructed_final_video.mp4'

# Get the list of frame filenames
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

# Check if there are frames to process
if not frame_files:
    print("No frames found in the specified directory.")
else:
    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec
    output_video = cv2.VideoWriter(output_video_path, fourcc, 10.0, (width, height))

    # Iterate through frames and write to video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        output_video.write(frame)

    # Release the VideoWriter object
    output_video.release()

    print(f"Video created successfully at {output_video_path}")
