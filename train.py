import pickle
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Configuration
LIDAR_MAX_RANGE_MM: int = 30000
NUM_RAYS: int = 200


class PalletPoseCNN(nn.Module):
    """Simple CNN for pallet pose estimation from LiDAR data."""

    def __init__(self, num_rays: int = NUM_RAYS) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            # Add dilated convolutions for larger receptive field
            nn.Conv1d(128, 128, 3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # x, y, theta
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_rays) -> (batch, 1, num_rays)
        x = x.unsqueeze(1)
        features = self.backbone(x).squeeze(-1)  # (batch, 256)
        return self.head(features)  # (batch, 3)


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the dataset."""
    with open("dataset.pkl", "rb") as f:
        data = pickle.load(f)

    # Extract data
    lidars = np.stack([d["lidar_data"] for d in data])
    positions = np.stack([d["distance"] for d in data])
    orientations = np.array([d["orientation"] for d in data])

    # Preprocess LiDAR data
    lidars = np.where(lidars < 0, LIDAR_MAX_RANGE_MM, lidars)
    lidars = lidars / LIDAR_MAX_RANGE_MM  # Normalize to [0,1]

    # Combine targets
    targets = np.column_stack([positions, orientations])

    return lidars.astype(np.float32), targets.astype(np.float32)


def create_dataloaders(
    lidars: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 512,
    val_split: float = 0.05,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    n_val = int(len(lidars) * val_split)

    # Split data
    val_lidars, val_targets = lidars[:n_val], targets[:n_val]
    train_lidars, train_targets = lidars[n_val:], targets[n_val:]

    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(train_lidars), torch.from_numpy(train_targets)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_lidars), torch.from_numpy(val_targets)
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def compute_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined position and orientation loss."""
    pred_pos, pred_angle = pred[:, :2], pred[:, 2]
    true_pos, true_angle = target[:, :2], target[:, 2]

    # Position loss (normalized by max range)
    pos_loss = nn.functional.mse_loss(pred_pos, true_pos) / LIDAR_MAX_RANGE_MM

    # Orientation loss (simple MSE in radians)
    angle_loss = nn.functional.mse_loss(pred_angle, true_angle)

    return pos_loss + angle_loss, pos_loss, angle_loss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = pos_loss = angle_loss = 0.0

    for batch_lidars, batch_targets in train_loader:
        batch_lidars, batch_targets = batch_lidars.to(device), batch_targets.to(device)

        optimizer.zero_grad()
        pred = model(batch_lidars)
        loss, p_loss, a_loss = compute_loss(pred, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pos_loss += p_loss.item()
        angle_loss += a_loss.item()

    n_batches = len(train_loader)
    return total_loss / n_batches, pos_loss / n_batches, angle_loss / n_batches


def validate(
    model: nn.Module, val_loader: DataLoader, device: torch.device
) -> Tuple[float, float, float]:
    """Validate the model."""
    model.eval()
    total_loss = pos_loss = angle_loss = 0.0

    with torch.no_grad():
        for batch_lidars, batch_targets in val_loader:
            batch_lidars, batch_targets = (
                batch_lidars.to(device),
                batch_targets.to(device),
            )

            pred = model(batch_lidars)
            loss, p_loss, a_loss = compute_loss(pred, batch_targets)

            total_loss += loss.item()
            pos_loss += p_loss.item()
            angle_loss += a_loss.item()

    n_batches = len(val_loader)
    return total_loss / n_batches, pos_loss / n_batches, angle_loss / n_batches


def main() -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading dataset...")
    lidars, targets = load_data()
    train_loader, val_loader = create_dataloaders(lidars, targets)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Model setup
    model = PalletPoseCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Try loading existing model
    # try:
    #     model.load_state_dict(torch.load("pallet_pose_cnn.pth", map_location=device))
    #     print("Loaded existing model")
    # except FileNotFoundError:
    #     print("Starting fresh training")

    # Training loop
    num_epochs = 50
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_pos, train_angle = train_epoch(
            model, train_loader, optimizer, device
        )

        # Validate
        val_loss, val_pos, val_angle = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step(val_loss)

        # Convert to interpretable units
        train_pos_mm = train_pos * LIDAR_MAX_RANGE_MM
        train_angle_deg = np.degrees(train_angle)
        val_pos_mm = val_pos * LIDAR_MAX_RANGE_MM
        val_angle_deg = np.degrees(val_angle)

        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"Train: {train_loss:.4f} ({train_pos_mm:.1f}mm, {train_angle_deg:.1f}°) | "
            f"Val: {val_loss:.4f} ({val_pos_mm:.1f}mm, {val_angle_deg:.1f}°)"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "pallet_pose_cnn.pth")

    print("Training complete!")


if __name__ == "__main__":
    main()
