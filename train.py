import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration (must match how dataset was generated)
LIDAR_MAX_RANGE_MM = 30000  # same as in data gen
NUM_RAYS = 200


def pallet_orientation_loss(pred_angles, true_angles):
    """
    Custom loss function for pallet orientation in the [0, π] range.

    Since the data generation now ensures orientations are in [0, π] and handles
    pallet symmetry, we can use a simpler loss function.

    Args:
        pred_angles: predicted angles in radians, shape (batch_size,)
        true_angles: true angles in radians, shape (batch_size,)

    Returns:
        loss: scalar tensor
    """
    # Calculate angular difference
    diff = pred_angles - true_angles

    # For angles in [0, π], we need to handle wraparound at the boundaries
    # Since pallets are symmetric, angles near 0 and π represent similar orientations
    abs_diff = torch.abs(diff)
    
    # Handle wraparound: if |diff| > π/2, use π - |diff| instead
    # This accounts for the symmetry where 0° and 180° are equivalent for pallets
    wrapped_diff = torch.where(abs_diff > np.pi/2, np.pi - abs_diff, abs_diff)
    
    # Return mean squared error
    return torch.mean(wrapped_diff**2)


# 4) Define the model
class PalletPoseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            # Add dilated convolutions for larger receptive field
            nn.Conv1d(128, 128, 3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (batch, 256, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Outputs x, y, theta
        )

    def forward(self, x):
        # x: (batch, NUM_RAYS)
        x = x.unsqueeze(1)  # → (batch,1,NUM_RAYS)
        f = self.features(x).squeeze(-1)  # → (batch,256)
        out = self.head(f)  # → (batch,3)

        # Apply appropriate activations:
        # - No activation on position outputs (x, y) - let the network learn the appropriate range
        # - For orientation: use modulo to constrain to [0, π] range
        out_pos = out[:, :2]  # Position outputs (x, y)
        
        # Use a different approach for orientation - let the network output raw values
        # and then use modulo to wrap them into [0, π] range
        # This avoids sigmoid saturation issues
        out_orient_raw = out[:, 2:3]
        # out_orient = torch.tanh(out_orient_raw) * (np.pi / 2.0) + (np.pi / 2.0)

        return torch.cat([out_pos, out_orient_raw], dim=1)


def main():
    # 1) Load the data
    with open("dataset.pkl", "rb") as f:
        data_list = pickle.load(f)

    # 2) Extract and preprocess into NumPy arrays
    #    - lidar_data: (N, NUM_RAYS)
    #    - positions: (N, 2)
    #    - orientations: (N,)
    lidars = np.stack([d["lidar_data"] for d in data_list], axis=0).astype(np.float32)
    positions = np.stack([d["distance"] for d in data_list], axis=0).astype(np.float32)
    orients = np.array([d["orientation"] for d in data_list], dtype=np.float32)

    # Replace "no hit" (-1) with max range, then normalize to [0,1]
    lidars[lidars < 0] = LIDAR_MAX_RANGE_MM
    lidars /= LIDAR_MAX_RANGE_MM  # now in [0,1]

    # Build targets: [x, y, theta]
    targets = np.stack([positions[:, 0], positions[:, 1], orients], axis=1).astype(
        np.float32
    )

    # 3) Move everything to GPU once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lidars_t = torch.from_numpy(lidars).to(device)  # shape (N, 200)
    targets_t = torch.from_numpy(targets).to(device)  # shape (N, 3)
    N = lidars_t.shape[0]

    model = PalletPoseCNN().to(device)

    # # Try to load existing model and continue training
    # try:
    #     model.load_state_dict(torch.load("pallet_pose_cnn.pth"))
    #     print("Loaded existing model, continuing training...")
    # except FileNotFoundError:
    #     print("No existing model found, starting fresh training...")

    model.train()

    # 5) Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    position_criterion = nn.MSELoss()
    orient_criterion = nn.MSELoss()
    batch_size = 512
    num_epochs = 100

    print(f"Training on {device}, N={N} samples")

    # Calculate and print the loss scaling factor
    degree_to_mm_scale = (100.0 / LIDAR_MAX_RANGE_MM) / (np.pi / 180.0)
    print(
        f"Loss scaling: 1 degree = 100mm, orientation loss scale factor = {degree_to_mm_scale:.4f}"
    )

    # 6) Training loop (with noise augmentation)
    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        running_pos_loss = 0.0
        running_orient_loss = 0.0
        running_total_loss = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            batch_in = lidars_t[idx]
            batch_tgt = targets_t[idx]

            # --- Data augmentation: small Gaussian noise on the ranges ---
            # noise = 0.01 * torch.randn_like(batch_in)
            #   batch_in = torch.clamp(batch_in + noise, 0.0, 1.0)

            pred = model(batch_in)

            # Split predictions and targets
            pred_pos = pred[:, :2]  # x, y
            pred_angle = pred[:, 2]  # theta

            true_pos = batch_tgt[:, :2]  # x, y
            true_angle = batch_tgt[:, 2]  # theta

            # print(f"pred_angle: {pred_angle}")
            # print(f"true_angle: {true_angle}")

            # Calculate losses
            pos_loss = position_criterion(pred_pos, true_pos) / LIDAR_MAX_RANGE_MM
            orient_loss = orient_criterion(pred_angle, true_angle)

            # Combine losses with balanced scaling
            # Use pre-calculated scaling factor
            total_loss = pos_loss + orient_loss
            # total_loss = orient_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_pos_loss += pos_loss.item() * batch_in.size(0)
            running_orient_loss += orient_loss.item() * batch_in.size(0)
            running_total_loss += total_loss.item() * batch_in.size(0)

        avg_pos_loss = running_pos_loss / N
        avg_orient_loss = running_orient_loss / N
        avg_total_loss = running_total_loss / N

        # Convert losses to interpretable units
        pos_accuracy_mm = avg_pos_loss * LIDAR_MAX_RANGE_MM
        orient_accuracy_deg = np.degrees(avg_orient_loss)

        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:02d}/{num_epochs} — Total: {avg_total_loss:.6f}, Pos: {avg_pos_loss:.6f} ({pos_accuracy_mm:.1f}mm), Orient: {avg_orient_loss:.6f} ({orient_accuracy_deg:.1f}°), LR: {current_lr:.2e}"
        )

        # Step the scheduler with the total loss
        scheduler.step(avg_total_loss)

    # 7) Save the trained model
    torch.save(model.state_dict(), "pallet_pose_cnn.pth")
    print("Training complete, model saved to pallet_pose_cnn.pth")


if __name__ == "__main__":
    main()
