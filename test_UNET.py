import os
import torch
import torch.nn as nn
import torch.utils.data as tdata
import numpy as np
import matplotlib.pyplot as plt
from dataloader import MyDataset  # Ensure this file is in your Python path
from training_with_pytorch import UNet  # Import your UNet model definition
from scipy.spatial.distance import directed_hausdorff

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set directories
data_dir = r"C:\DISSERTATION\Data for Project\Data_by_modality\VOXEL_SIZED_FOLDER"
model_path = "./result/checkpoint_step_98.pth"  # Path to your best/desired checkpoint
output_dir = "./test_dataset_predictions"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------------------
# Load Test Dataset
# -------------------------------------------------------------------------
print("Initializing test dataset...")
test_set = MyDataset(data_dir, 'test')
test_loader = tdata.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
print(f"Test dataset initialized. Size: {len(test_set)}")

# -------------------------------------------------------------------------
# Load Model
# -------------------------------------------------------------------------
print(f"Loading model from {model_path}...")
model = UNet(ch_in=3, ch_out=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully.")

# -------------------------------------------------------------------------
# Dice Score Function
# -------------------------------------------------------------------------
def compute_dice_score(y_pred, y_true, eps=1e-6):
    """
    Compute Dice Score.
    """
    y_pred = (y_pred > 0.5).float()  # Threshold to binary
    numerator = torch.sum(y_true * y_pred, dim=(2, 3, 4)) * 2
    denominator = torch.sum(y_true, dim=(2, 3, 4)) + torch.sum(y_pred, dim=(2, 3, 4)) + eps
    dice_score = numerator / denominator
    return dice_score.mean().item()

# -------------------------------------------------------------------------
# Hausdorff Distance Function
# -------------------------------------------------------------------------
def compute_hausdorff_distance(y_pred, y_true):
    """
    Compute Hausdorff Distance between predicted and true binary masks.
    """
    # Flatten the predictions and ground truth to 2D for easier computation
    y_pred_flat = (y_pred > 0.5).cpu().numpy().astype(np.uint8).flatten()
    y_true_flat = y_true.cpu().numpy().astype(np.uint8).flatten()

    # Find the coordinates of the non-zero values (the foreground)
    pred_coords = np.column_stack(np.where(y_pred_flat > 0))
    true_coords = np.column_stack(np.where(y_true_flat > 0))

    if len(pred_coords) == 0 or len(true_coords) == 0:
        return float('inf')  # Return infinity if there are no lesions in either prediction or ground truth

    # Compute the directed Hausdorff distance from predicted to true and vice versa
    hd_pred_to_true = directed_hausdorff(pred_coords, true_coords)[0]
    hd_true_to_pred = directed_hausdorff(true_coords, pred_coords)[0]

    # The Hausdorff distance is the maximum of the two directed distances
    return max(hd_pred_to_true, hd_true_to_pred)

# -------------------------------------------------------------------------
# Run Test and Save Predictions
# -------------------------------------------------------------------------
print("Starting inference on test set...")

dice_scores = []  # Store Dice scores for all test patients
hausdorff_distances = []  # Store Hausdorff distances for all test patients

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Extract patient ID (or other ID) from dataset
        patient_id = test_set.data_list[batch_idx][0]  # Example extraction

        # Forward pass (inference)
        preds = model(images)

        # Compute Dice Score
        dice_score_val = compute_dice_score(preds, labels)
        dice_scores.append(dice_score_val)

        # Compute Hausdorff Distance
        hausdorff_distance_val = compute_hausdorff_distance(preds, labels)
        hausdorff_distances.append(hausdorff_distance_val)

        # Convert tensors to numpy
        images_np = images.cpu().numpy()[0, 2]  # T2-weighted channel (index 2)
        labels_np = labels.cpu().numpy()[0, 0]  # Lesion ground truth
        preds_np = (preds.cpu().numpy()[0, 0] > 0.5).astype(np.uint8)

        # Identify slices with nonzero ground truth (for plotting)
        nonzero_slices = np.where(np.sum(labels_np, axis=(0, 1)) > 0)[0]
        if len(nonzero_slices) == 0:
            print(f"Skipping Patient {patient_id} (No lesions in ground truth).")
            continue

        # Create a figure of these slices
        num_slices = len(nonzero_slices)
        fig, axes = plt.subplots(num_slices, 4, figsize=(12, num_slices * 2.5))
        fig.suptitle(f"Patient {patient_id} - Dice Score: {dice_score_val:.4f}, Hausdorff Distance: {hausdorff_distance_val:.4f}", fontsize=16, y=1.02)

        for i, slice_idx in enumerate(nonzero_slices):
            axes[i, 0].imshow(images_np[:, :, slice_idx], cmap='gray')
            axes[i, 0].set_title(f"T2W - Slice {slice_idx}", fontsize=10)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(labels_np[:, :, slice_idx], cmap='jet')
            axes[i, 1].set_title(f"Ground Truth - Slice {slice_idx}", fontsize=10)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(preds_np[:, :, slice_idx], cmap='jet')
            axes[i, 2].set_title(f"Prediction - Slice {slice_idx}", fontsize=10)
            axes[i, 2].axis("off")

            # Overlay predicted mask on T2W
            axes[i, 3].imshow(images_np[:, :, slice_idx], cmap='gray')
            axes[i, 3].imshow(preds_np[:, :, slice_idx], cmap='jet', alpha=0.5)
            axes[i, 3].set_title("Overlay Prediction", fontsize=10)
            axes[i, 3].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Save figure
        output_filepath = os.path.join(output_dir, f"test_patient_{patient_id}.png")
        plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"[{batch_idx+1}/{len(test_loader)}] Patient {patient_id} | Dice: {dice_score_val:.4f} | Hausdorff: {hausdorff_distance_val:.4f} | Saved: {output_filepath}")

# Compute overall average Dice and Hausdorff Distance
mean_dice = np.mean(dice_scores) if len(dice_scores) else 0
mean_hausdorff = np.mean(hausdorff_distances) if len(hausdorff_distances) else float('inf')

print(f"\nAverage Dice Score on Test Set: {mean_dice:.4f}")
print(f"Average Hausdorff Distance on Test Set: {mean_hausdorff:.4f}")
print("Test set inference complete. Predictions saved.")
