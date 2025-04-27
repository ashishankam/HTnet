import os
import cv2
import numpy as np

# Function to calculate optical flow using TV-L1
def tvl1_ofcalc(path1, path2, epsilon=0.15):
    img1 = image_resizing(path1)
    img2 = image_resizing(path2)
    flow = cv2.optflow.createOptFlow_DualTVL1()
    flow.setEpsilon(epsilon)
    of = flow.calc(img1, img2, None)
    return of

def image_resizing(img_path):
    origin_photo1 = cv2.imread(img_path)
    resized_photo1 = cv2.resize(origin_photo1, (28, 28))
    img = cv2.cvtColor(resized_photo1, cv2.COLOR_BGR2GRAY)
    return img

# Function for min-max normalization of optical flow values
def minmax_norm(x):
    x_flat = x.reshape(-1)
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max == x_min:
        x_flat *= 0
    else:
        x_flat = (x_flat - x_min) / (x_max - x_min)
    return x_flat.reshape(x.shape)

# Function to calculate optical strain flow and concatenate results
def calc_os_flow(path1, path2, epsilon=0.15):
    flow = tvl1_ofcalc(path1, path2, epsilon)
    u_flow = minmax_norm(flow[:, :, 0]) * 255
    v_flow = minmax_norm(flow[:, :, 1]) * 255
    ux, uy = np.gradient(flow[:, :, 0])
    vx, vy = np.gradient(flow[:, :, 1])
    os_flow = np.sqrt(ux ** 2 + vy ** 2 + 0.25 * (uy + vx) ** 2)
    os_flow = minmax_norm(os_flow) * 255
    return np.concatenate((os_flow.reshape(*os_flow.shape, 1), v_flow.reshape(*v_flow.shape, 1), u_flow.reshape(*u_flow.shape, 1)), axis=2)

# Function to process folders and generate optical flow
def process_folders(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(image_folder):
        for directory in dirs:
            subject_folder = os.path.join(root, directory)
            image_files = sorted([f for f in os.listdir(subject_folder) if f.endswith('.jpg')])

            if len(image_files) != 2:
                print(f"Expected 2 images in {subject_folder}, found {len(image_files)}.")
                continue

            # First image is the onset, second image is the apex
            onset_image = os.path.join(subject_folder, image_files[0])
            apex_image = os.path.join(subject_folder, image_files[1])

            # Calculate optical flow
            optical_flow = calc_os_flow(onset_image, apex_image)

            # Extract the first part of the folder name to prefix the output file
            folder_prefix = directory.split('_')[0]
            new_file_name = f'{folder_prefix}_{directory}.png'

            # Save optical flow image with the modified name in the output folder
            save_path = os.path.join(output_folder, new_file_name)
            cv2.imwrite(save_path, optical_flow)
            print(f"Optical flow saved at {save_path}")

# Folder containing the original images
image_folder = r"C:\HTNet Final Project\Optical Flow --  TESTING\Optical Flow --  TESTING\Optical Flow\CASME 2\resized_filtered_frames"  # Update the path

# Folder where optical flow images will be saved
output_folder = r'C:\HTNet Final Project\Optical Flow --  TESTING\Optical Flow --  TESTING\Optical Flow\CASME 2\optical_flow_images'  # Update the path

# Run the process
process_folders(image_folder, output_folder)


# # Folder containing the original images
# image_folder = r"C:\HTNet Final Project\Optical Flow\samm\cropped-samm-onset-apex"  # Update the path

# # Folder where optical flow images will be saved
# output_folder = r'C:\HTNet Final Project\Optical Flow\samm\optical-samm'  # Update the path

# # Run the process
# process_folders(image_folder, output_folder)
