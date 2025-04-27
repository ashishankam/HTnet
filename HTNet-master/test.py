import os
import cv2
import numpy as np
import torch
from Model import HTNet  # Assuming the HTNet model is in Model.py

# Function to calculate optical flow using TV-L1
def tvl1_ofcalc(path1, path2, epsilon=0.15):
    origin_photo1 = cv2.imread(path1)
    origin_photo2 = cv2.imread(path2)
    resized_photo1 = cv2.resize(origin_photo1, (28, 28))
    resized_photo2 = cv2.resize(origin_photo2, (28, 28))
    img1 = cv2.cvtColor(resized_photo1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(resized_photo2, cv2.COLOR_BGR2GRAY)
    flow = cv2.optflow.createOptFlow_DualTVL1()
    flow.setEpsilon(epsilon)
    of = flow.calc(img1, img2, None)
    return of

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

# Function to test an unseen image
def test_unseen_image(onset_image_path=r"E:\HTnet\onset.jpg", apex_image_path=r"E:\HTnet\apex.jpg", weights_folder=r"E:\HTnet\HTNet-master\ourmodel_threedatasets_weights"):
    """
    Test an unseen image pair against all weights and average the predictions.
    Args:
        onset_image_path: Path to the onset image.
        apex_image_path: Path to the apex image.
        weights_folder: Folder containing .pth weight files.
        model_config: Model configuration for initializing HTNet.
    Returns:
        Averaged prediction as a class index.
    """
    model_config = {
    "image_size": 28,
    "patch_size": 7,
    "dim": 256,
    "heads": 3,
    "num_hierarchies": 3,
    "block_repeats": (2, 2, 10),
    "num_classes": 3  # Adjust according to the training setup
    }
    # Generate optical flow for the unseen input
    optical_flow = calc_os_flow(onset_image_path, apex_image_path)
    optical_flow_resized = cv2.resize(optical_flow, (28, 28))  # Ensure size matches model input
    optical_flow_tensor = torch.Tensor(optical_flow_resized).permute(2, 0, 1).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    cv2.imwrite("Optical_Flow_Image.png", optical_flow_resized)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Close the image window

    # Initialize the model
    model = HTNet(**model_config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Load all weights and test
    predictions = []
    for weight_file in os.listdir(weights_folder):
        if weight_file.endswith('.pth'):
            weight_path = os.path.join(weights_folder, weight_file)
            model.load_state_dict(torch.load(weight_path))
            with torch.no_grad():
                output = model(optical_flow_tensor)
                pred = torch.argmax(output, dim=1).item()
                predictions.append(pred)

    # Return the averaged prediction
    average_prediction = max(set(predictions), key=predictions.count)  # Majority vote
    print(f"Averaged Prediction: {average_prediction}")
    return average_prediction

# Paths for the unseen image pair


# Folder containing weights
# weights_folder = r"E:\HTnet\HTNet-master\ourmodel_threedatasets_weights"  # Update the path

# Model configuration (same as used in main_HTNet.py)
# model_config = {
#     "image_size": 28,
#     "patch_size": 7,
#     "dim": 256,
#     "heads": 3,
#     "num_hierarchies": 3,
#     "block_repeats": (2, 2, 10),
#     "num_classes": 3  # Adjust according to the training setup
# }

# Run the test
# test_unseen_image()
# test_unseen_image(onset_image_path, apex_image_path, weights_folder, model_config)
