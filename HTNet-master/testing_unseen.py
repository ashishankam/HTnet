import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import HTNet  # Import the model class

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
    x_max, x_min = np.max(x), np.min(x)
    return np.zeros_like(x) if x_max == x_min else (x - x_min) / (x_max - x_min)

# Function to compute the optical strain flow and concatenate results
def calc_os_flow(path1, path2, epsilon=0.15):
    flow = tvl1_ofcalc(path1, path2, epsilon)
    u_flow = minmax_norm(flow[:, :, 0]) * 255
    v_flow = minmax_norm(flow[:, :, 1]) * 255
    ux, uy = np.gradient(flow[:, :, 0])
    vx, vy = np.gradient(flow[:, :, 1])
    os_flow = np.sqrt(ux**2 + vy**2 + 0.25 * (uy + vx)**2)
    os_flow = minmax_norm(os_flow) * 255
    return np.stack((os_flow, v_flow, u_flow), axis=2)

# Define a simple feature extractor (for example, a small CNN)
class FeatureExtractor(nn.Module):
    def __init__(self, output_dim=3):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, 128)  # Adjust size based on input image size (28x28)
        self.fc2 = nn.Linear(128, output_dim)  # Output to match the model output (e.g., number of classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc(x))
        x = self.fc2(x)  # Output to match the model output size
        return x

# Function to calculate cosine similarity
def cosine_similarity(tensor1, tensor2):
    tensor1_flat = tensor1.view(-1)  # Flatten the optical flow tensor to 1D
    tensor2_flat = tensor2.view(-1)  # Flatten the model's extracted features to 1D
    print(f"Tensor1 shape: {tensor1_flat.shape}, Tensor2 shape: {tensor2_flat.shape}")  # Debug print to check sizes
    return F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0), dim=1).item()

# Modify the test function to ensure correct tensor shapes
def test_unseen_image(onset_path, apex_path, weights_folder, model_config):
    """
    Test an unseen image pair against the best matching weight file.
    """
    # Generate optical flow for the unseen input
    optical_flow = calc_os_flow(onset_path, apex_path)
    optical_flow_resized = cv2.resize(optical_flow, (28, 28))  # Ensure size matches model input
    optical_flow_tensor = torch.Tensor(optical_flow_resized).permute(2, 0, 1).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(output_dim=model_config["num_classes"]).to("cuda" if torch.cuda.is_available() else "cpu")

    # Extract features from the optical flow tensor
    optical_flow_features = feature_extractor(optical_flow_tensor)

    # Print the shape of the extracted features
    print(f"Extracted features shape: {optical_flow_features.shape}")

    # Initialize the model
    model = HTNet(**model_config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set model to evaluation mode

    best_similarity = -1  # Store highest similarity
    best_weight_file = None  # Store best matching weight file

    # Compare unseen optical flow with trained weights using Cosine Similarity
    for weight_file in os.listdir(weights_folder):
        if weight_file.endswith('.pth'):
            weight_path = os.path.join(weights_folder, weight_file)
            model.load_state_dict(torch.load(weight_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))

            with torch.no_grad():
                output = model(optical_flow_tensor)
                # Flatten the model output to make it comparable
                model_features = output.view(1, -1)  # Flatten model output to a 1D tensor

                # Print the shape of the model output
                print(f"Model output shape: {model_features.shape}")

                # Compare features using cosine similarity
                similarity = cosine_similarity(optical_flow_features, model_features)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_weight_file = weight_path  # Store best matching weight file

    # Load the best matching weight file and make final prediction
    if best_weight_file:
        model.load_state_dict(torch.load(best_weight_file, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            output = model(optical_flow_tensor)
            pred = torch.argmax(F.softmax(output, dim=1), dim=1).item()

        print(f"🔹 Best Matching Weights: {best_weight_file}")
        print(f"✅ Final Prediction: {pred} (Class)")
        return pred
    else:
        print("❌ No suitable weight file found!")
        return None

# Paths for the unseen image pair
onset_image_path = r"E:\HTnet\onset.jpg"  # Update the path
apex_image_path = r"E:\HTnet\apex.jpg"  # Update the path

# Folder containing weights
weights_folder = r"E:\HTnet\HTNet-master\ourmodel_threedatasets_weights"  # Update the path

# Model configuration (same as used in main_HTNet.py)
model_config = {
    "image_size": 28,
    "patch_size": 7,
    "dim": 256,
    "heads": 3,
    "num_hierarchies": 3,
    "block_repeats": (2, 2, 10),
    "num_classes": 3  # Adjust according to the training setup
}

# Run the test
# test_unseen_image(onset_image_path, apex_image_path, weights_folder, model_config = model_config)
