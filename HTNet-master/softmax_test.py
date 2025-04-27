import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os

# Optical flow generation functions
def tvl1_ofcalc(path1, path2, epsilon=0.15):
    origin_photo1 = cv2.imread(path1)
    origin_photo2 = cv2.imread(path2)
    resized_photo1 = cv2.resize(origin_photo1, (28, 28))
    resized_photo2 = cv2.resize(origin_photo2, (28, 28))
    img1 = cv2.cvtColor(resized_photo1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(resized_photo2, cv2.COLOR_BGR2GRAY)
    flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow.setEpsilon(epsilon)
    of = flow.calc(img1, img2, None)
    return of

def minmax_norm(x):
    x_flat = x.reshape(-1)
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max == x_min:
        x_flat *= 0
    else:
        x_flat = (x_flat - x_min) / (x_max - x_min)
    return x_flat.reshape(x.shape)

def calc_os_flow(path1, path2, epsilon=0.15):
    flow = tvl1_ofcalc(path1, path2, epsilon)
    u_flow = minmax_norm(flow[:, :, 0]) * 255
    v_flow = minmax_norm(flow[:, :, 1]) * 255
    ux, uy = np.gradient(flow[:, :, 0])
    vx, vy = np.gradient(flow[:, :, 1])
    os_flow = np.sqrt(ux ** 2 + vy ** 2 + 0.25 * (uy + vx) ** 2)
    os_flow = minmax_norm(os_flow) * 255
    return np.concatenate((os_flow.reshape(*os_flow.shape, 1), 
                           v_flow.reshape(*v_flow.shape, 1), 
                           u_flow.reshape(*u_flow.shape, 1)), axis=2)

# Softmax Averaging Implementation
def test_unseen_image(onset_image_path, apex_image_path, weights_folder, model_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load optical flow from the unseen images
    optical_flow = calc_os_flow(onset_image_path, apex_image_path)
    optical_flow_resized = cv2.resize(optical_flow, (28, 28))  # Resize to match input size
    optical_flow_tensor = torch.Tensor(optical_flow_resized).permute(2, 0, 1).unsqueeze(0).to(device)

    # Initialize model
    from Model import HTNet  # Assuming HTNet is defined in the Model module
    model = HTNet(
        image_size=28,
        patch_size=7,
        dim=model_config['dim'],
        heads=model_config['heads'],
        num_hierarchies=model_config['num_hierarchies'],
        block_repeats=model_config['block_repeats'],
        num_classes=3  # Change according to your use case
    )
    model.to(device)
    
    # Get all weight files in the folder
    weight_files = [f for f in os.listdir(weights_folder) if f.endswith('.pth')]
    if not weight_files:
        raise ValueError("No weight files found in the specified folder.")

    # Accumulate softmax probabilities
    avg_probabilities = None
    for weight_file in weight_files:
        weight_path = os.path.join(weights_folder, weight_file)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        with torch.no_grad():
            output = model(optical_flow_tensor)
            softmax_probs = F.softmax(output, dim=1).cpu().numpy()

            if avg_probabilities is None:
                avg_probabilities = softmax_probs
            else:
                avg_probabilities += softmax_probs

    # Average the probabilities
    avg_probabilities /= len(weight_files)

    # Final prediction
    final_prediction = np.argmax(avg_probabilities, axis=1)[0]
    print(f"Final Prediction: {final_prediction}")
    print(f"Softmax Probabilities: {avg_probabilities}")
    return final_prediction

# Paths to the images and weights folder
onset_image_path = r'E:\HTnet\onset.jpg'  # Update with the actual path
apex_image_path = r'E:\HTnet\apex.jpg'  # Update with the actual path
weights_folder = r'E:\HTnet\HTNet-master\head4_ourmodel_threedatasets_weights'  # Update with the actual path

# Model configuration
model_config = {
    'dim': 256,
    'heads': 4,
    'num_hierarchies': 3,
    'block_repeats': (2, 2, 10)
}

# Run the test
test_unseen_image(onset_image_path, apex_image_path, weights_folder, model_config)
