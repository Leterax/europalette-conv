import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
import time
from train import PalletPoseCNN

# Configuration (must match training script)
LIDAR_MAX_RANGE_MM = 30000
NUM_RAYS = 200


def decode_model_output(output):
    """
    Decode the model output [x, y, theta] to position and orientation.

    Args:
        output: torch.Tensor or numpy array of shape (3,) containing [x, y, theta]

    Returns:
        position: numpy array [x, y] in mm
        orientation: float, orientation angle in radians in [0, π] range
    """
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    x, y, theta = output

    position = np.array([x, y])
    # The model now outputs orientations in [0, π] range, so we just clamp to ensure bounds
    orientation = np.clip(theta, 0, np.pi)

    return position, orientation


def calculate_errors(pred_pos, pred_orient, true_pos, true_orient):
    """Calculate position and orientation errors."""
    # Position error (Euclidean distance)
    pos_error = np.linalg.norm(pred_pos - true_pos)

    # Orientation error for angles in [0, π] range
    # Since pallets are symmetric, orientations θ and θ+π are equivalent
    # In the [0, π] range, this means angles near 0 and π represent the same orientation

    # Calculate direct difference
    abs_diff = np.abs(pred_orient - true_orient)

    # Calculate alternative difference considering that 0 and π are equivalent
    # (due to pallet symmetry within the [0, π] range)
    alt_diff = np.pi - abs_diff

    # Take the minimum error
    orient_error = min(abs_diff, alt_diff)

    return pos_error, orient_error


def draw_pallet_at_position(
    ax, position, orientation, color="red", alpha=0.7, label=None
):
    """
    Draw a simplified pallet representation at the given position and orientation.

    Args:
        ax: matplotlib axis
        position: [x, y] position in mm
        orientation: orientation angle in radians
        color: color for the pallet
        alpha: transparency
        label: label for legend
    """
    # Pallet dimensions (approximate europalette)
    pallet_width = 1200  # mm
    pallet_height = 800  # mm

    # Create rectangle centered at origin
    rect_x = -pallet_width / 2
    rect_y = -pallet_height / 2

    # Create the rectangle
    rect = Rectangle(
        (rect_x, rect_y),
        pallet_width,
        pallet_height,
        angle=np.degrees(orientation),
        rotation_point="center",
        facecolor=color,
        alpha=alpha,
        edgecolor="black",
        linewidth=2,
    )

    # Transform to position
    t = (
        plt.matplotlib.transforms.Affine2D().translate(position[0], position[1])
        + ax.transData
    )
    rect.set_transform(t)

    ax.add_patch(rect)

    # Add orientation arrow
    arrow_length = 300  # mm
    arrow_start = position
    arrow_end = position + arrow_length * np.array(
        [np.cos(orientation), np.sin(orientation)]
    )

    arrow = FancyArrowPatch(
        arrow_start,
        arrow_end,
        arrowstyle="->",
        mutation_scale=20,
        color=color,
        linewidth=3,
    )
    ax.add_patch(arrow)

    # Add label if provided
    if label:
        ax.text(
            position[0],
            position[1] + pallet_height / 2 + 100,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
            fontweight="bold",
        )


def plot_evaluation_result(
    lidar_data,
    true_pos,
    true_orient,
    pred_pos,
    pred_orient,
    pos_error,
    orient_error,
    sample_idx,
):
    """Plot the evaluation result with lidar data and pallet positions."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Raw lidar data
    ax1.plot(lidar_data, "b-", alpha=0.7, linewidth=1)
    ax1.set_title(f"Sample {sample_idx}: Raw LiDAR Data")
    ax1.set_xlabel("Ray Index")
    ax1.set_ylabel("Distance (mm)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, LIDAR_MAX_RANGE_MM)

    # Add some statistics
    valid_readings = lidar_data[lidar_data > 0]
    if len(valid_readings) > 0:
        ax1.axhline(
            np.mean(valid_readings),
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {np.mean(valid_readings):.0f}mm",
        )
        ax1.axhline(
            np.min(valid_readings),
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Min: {np.min(valid_readings):.0f}mm",
        )
        ax1.legend()

    # Plot 2: Spatial view with predicted and true pallet positions
    # Calculate the view bounds based on the data
    max_range = max(np.max(true_pos), np.max(pred_pos), 2000)
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)

    # Draw origin (sensor position)
    ax2.scatter(0, 0, c="black", s=100, marker="o", label="Sensor Origin", zorder=5)

    # Draw true pallet position
    draw_pallet_at_position(
        ax2, true_pos, true_orient, color="green", alpha=0.5, label="True Pallet"
    )

    # Draw predicted pallet position
    draw_pallet_at_position(
        ax2, pred_pos, pred_orient, color="red", alpha=0.5, label="Predicted Pallet"
    )

    # Draw lidar rays and hits
    angles = np.linspace(-np.pi * 3 / 4, np.pi * 3 / 4, NUM_RAYS)  # 270° FOV

    # Sample every 50th ray for visualization
    sample_indices = np.arange(0, len(angles), 50)
    for i in sample_indices:
        angle = angles[i]
        distance = lidar_data[i]

        ray_dir = np.array([np.cos(angle), np.sin(angle)])

        if distance > 0 and distance < LIDAR_MAX_RANGE_MM:
            # Ray hits something
            hit_point = distance * ray_dir
            ax2.plot(
                [0, hit_point[0]], [0, hit_point[1]], "g-", alpha=0.3, linewidth=0.5
            )
            ax2.scatter(hit_point[0], hit_point[1], c="blue", s=1, alpha=0.6)
        else:
            # Ray doesn't hit - draw shorter line
            end_point = 1000 * ray_dir
            ax2.plot(
                [0, end_point[0]], [0, end_point[1]], "r-", alpha=0.2, linewidth=0.5
            )

    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.legend()

    # Add error information to the title
    title = f"Sample {sample_idx}: Pallet Position Estimation\n"
    title += f"Position Error: {pos_error:.1f}mm, Orientation Error: {np.degrees(orient_error):.1f}°"
    ax2.set_title(title)

    plt.tight_layout()
    return fig


def main():
    print("Loading trained model and dataset...")

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalletPoseCNN().to(device)

    try:
        model.load_state_dict(torch.load("pallet_pose_cnn.pth", map_location=device))
        model.eval()
        print(f"Model loaded successfully on {device}")
    except FileNotFoundError:
        print("Error: pallet_pose_cnn.pth not found. Please run train.py first.")
        return

    # Load the dataset
    try:
        with open("dataset.pkl", "rb") as f:
            data_list = pickle.load(f)
        print(f"Dataset loaded: {len(data_list)} samples")
    except FileNotFoundError:
        print("Error: dataset.pkl not found. Please run generate_data.py first.")
        return

    # Prepare data (same preprocessing as in training)
    lidars = np.stack([d["lidar_data"] for d in data_list], axis=0).astype(np.float32)
    positions = np.stack([d["distance"] for d in data_list], axis=0).astype(np.float32)
    orientations = np.array([d["orientation"] for d in data_list], dtype=np.float32)

    # Preprocess lidar data
    lidars[lidars < 0] = LIDAR_MAX_RANGE_MM
    lidars_normalized = lidars / LIDAR_MAX_RANGE_MM

    # Convert to torch tensors
    lidars_t = torch.from_numpy(lidars_normalized).to(device)

    print("\nRunning inference on dataset...")
    start_time = time.time()

    # Run inference
    with torch.no_grad():
        predictions = model(lidars_t)

    inference_time = time.time() - start_time
    print(
        f"Inference completed in {inference_time:.3f}s ({len(data_list) / inference_time:.1f} samples/sec)"
    )

    # Add diagnostic checks for prediction diversity
    print("\n" + "=" * 60)
    print("PREDICTION DIVERSITY ANALYSIS")
    print("=" * 60)

    # Check raw prediction statistics
    raw_predictions = predictions.cpu().numpy()
    print("Raw model outputs (first 10 samples):")
    for i in range(min(10, len(raw_predictions))):
        print(
            f"  Sample {i}: [{raw_predictions[i, 0]:.3f}, {raw_predictions[i, 1]:.3f}, {raw_predictions[i, 2]:.3f}]"
        )

    print("\nRaw prediction statistics:")
    for j, name in enumerate(["x", "y", "theta"]):
        values = raw_predictions[:, j]
        print(
            f"  {name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, min={np.min(values):.3f}, max={np.max(values):.3f}"
        )

    # Check if predictions are too similar
    pred_std = np.std(raw_predictions, axis=0)
    print(f"\nPrediction standard deviations: {pred_std}")
    if np.all(pred_std < 0.1):
        print(
            "WARNING: Very low prediction diversity - model might be predicting similar values!"
        )

    # Plot prediction distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = ["X Position", "Y Position", "Theta (radians)"]

    for i in range(3):
        axes[i].hist(raw_predictions[:, i], bins=50, alpha=0.7, edgecolor="black")
        axes[i].set_title(f"{names[i]} Distribution")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(
            np.mean(raw_predictions[:, i]),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(raw_predictions[:, i]):.3f}",
        )
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    # Calculate errors for all samples
    pos_errors = []
    orient_errors = []

    print("\nCalculating errors...")
    for i in range(len(data_list)):
        # Decode prediction
        pred_pos, pred_orient = decode_model_output(predictions[i])

        # True values
        true_pos = positions[i]
        true_orient = orientations[i]

        # Calculate errors
        pos_error, orient_error = calculate_errors(
            pred_pos, pred_orient, true_pos, true_orient
        )
        pos_errors.append(pos_error)
        orient_errors.append(orient_error)

    pos_errors = np.array(pos_errors)
    orient_errors = np.array(orient_errors)

    # Add comparison between ground truth and predictions
    print("\n" + "=" * 60)
    print("GROUND TRUTH vs PREDICTIONS COMPARISON")
    print("=" * 60)

    # Extract predictions
    pred_positions = raw_predictions[:, :2]
    pred_orientations = raw_predictions[:, 2]
    # Predicted orientations are already in [0, π] range from the model

    print("Ground Truth Statistics:")
    print(
        f"  X position: mean={np.mean(positions[:, 0]):.1f}, std={np.std(positions[:, 0]):.1f}"
    )
    print(
        f"  Y position: mean={np.mean(positions[:, 1]):.1f}, std={np.std(positions[:, 1]):.1f}"
    )
    print(
        f"  Orientation [0, π]: mean={np.mean(orientations):.3f}, std={np.std(orientations):.3f}"
    )

    print("\nPredicted Statistics:")
    print(
        f"  X position: mean={np.mean(pred_positions[:, 0]):.1f}, std={np.std(pred_positions[:, 0]):.1f}"
    )
    print(
        f"  Y position: mean={np.mean(pred_positions[:, 1]):.1f}, std={np.std(pred_positions[:, 1]):.1f}"
    )
    print(
        f"  Orientation [0, π]: mean={np.mean(pred_orientations):.3f}, std={np.std(pred_orientations):.3f}"
    )

    # Plot ground truth vs predictions comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # X position comparison
    axes[0, 0].hist(
        positions[:, 0], bins=50, alpha=0.5, label="Ground Truth", color="green"
    )
    axes[0, 0].hist(
        pred_positions[:, 0], bins=50, alpha=0.5, label="Predicted", color="red"
    )
    axes[0, 0].set_title("X Position Distribution")
    axes[0, 0].set_xlabel("X (mm)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Y position comparison
    axes[0, 1].hist(
        positions[:, 1], bins=50, alpha=0.5, label="Ground Truth", color="green"
    )
    axes[0, 1].hist(
        pred_positions[:, 1], bins=50, alpha=0.5, label="Predicted", color="red"
    )
    axes[0, 1].set_title("Y Position Distribution")
    axes[0, 1].set_xlabel("Y (mm)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Orientation comparison
    axes[0, 2].hist(
        orientations, bins=50, alpha=0.5, label="Ground Truth", color="green"
    )
    axes[0, 2].hist(
        pred_orientations, bins=50, alpha=0.5, label="Predicted", color="red"
    )
    axes[0, 2].set_title("Orientation Distribution")
    axes[0, 2].set_xlabel("Theta (radians)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Scatter plots
    axes[1, 0].scatter(positions[:, 0], pred_positions[:, 0], alpha=0.3, s=1)
    axes[1, 0].plot(
        [np.min(positions[:, 0]), np.max(positions[:, 0])],
        [np.min(positions[:, 0]), np.max(positions[:, 0])],
        "r--",
        label="Perfect prediction",
    )
    axes[1, 0].set_xlabel("True X (mm)")
    axes[1, 0].set_ylabel("Predicted X (mm)")
    axes[1, 0].set_title("X Position: True vs Predicted")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(positions[:, 1], pred_positions[:, 1], alpha=0.3, s=1)
    axes[1, 1].plot(
        [np.min(positions[:, 1]), np.max(positions[:, 1])],
        [np.min(positions[:, 1]), np.max(positions[:, 1])],
        "r--",
        label="Perfect prediction",
    )
    axes[1, 1].set_xlabel("True Y (mm)")
    axes[1, 1].set_ylabel("Predicted Y (mm)")
    axes[1, 1].set_title("Y Position: True vs Predicted")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].scatter(orientations, pred_orientations, alpha=0.3, s=1)
    axes[1, 2].plot(
        [np.min(orientations), np.max(orientations)],
        [np.min(orientations), np.max(orientations)],
        "r--",
        label="Perfect prediction",
    )
    axes[1, 2].set_xlabel("True Theta (radians)")
    axes[1, 2].set_ylabel("Predicted Theta (radians)")
    axes[1, 2].set_title("Orientation: True vs Predicted")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print overall statistics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset size: {len(data_list)} samples")
    print(
        f"Inference time: {inference_time:.3f}s ({len(data_list) / inference_time:.1f} samples/sec)"
    )
    print("\nPosition Error Statistics:")
    print(f"  Mean: {np.mean(pos_errors):.1f}mm")
    print(f"  Median: {np.median(pos_errors):.1f}mm")
    print(f"  Std: {np.std(pos_errors):.1f}mm")
    print(f"  Min: {np.min(pos_errors):.1f}mm")
    print(f"  Max: {np.max(pos_errors):.1f}mm")
    print(f"  95th percentile: {np.percentile(pos_errors, 95):.1f}mm")

    print("\nOrientation Error Statistics:")
    print(f"  Mean: {np.degrees(np.mean(orient_errors)):.1f}°")
    print(f"  Median: {np.degrees(np.median(orient_errors)):.1f}°")
    print(f"  Std: {np.degrees(np.std(orient_errors)):.1f}°")
    print(f"  Min: {np.degrees(np.min(orient_errors)):.1f}°")
    print(f"  Max: {np.degrees(np.max(orient_errors)):.1f}°")
    print(f"  95th percentile: {np.degrees(np.percentile(orient_errors, 95)):.1f}°")

    # Accuracy thresholds
    pos_thresholds = [50, 100, 200, 500]  # mm
    orient_thresholds = [5, 10, 15, 30]  # degrees

    print("\nAccuracy Analysis:")
    print("Position accuracy:")
    for thresh in pos_thresholds:
        accuracy = np.mean(pos_errors < thresh) * 100
        print(f"  < {thresh}mm: {accuracy:.1f}%")

    print("Orientation accuracy:")
    for thresh in orient_thresholds:
        thresh_rad = np.radians(thresh)
        accuracy = np.mean(orient_errors < thresh_rad) * 100
        print(f"  < {thresh}°: {accuracy:.1f}%")

    # Plot error distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(pos_errors, bins=50, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Position Error (mm)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Position Error Distribution")
    ax1.axvline(
        np.mean(pos_errors),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(pos_errors):.1f}mm",
    )
    ax1.axvline(
        np.median(pos_errors),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(pos_errors):.1f}mm",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(np.degrees(orient_errors), bins=50, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Orientation Error (degrees)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Orientation Error Distribution")
    ax2.axvline(
        np.degrees(np.mean(orient_errors)),
        color="red",
        linestyle="--",
        label=f"Mean: {np.degrees(np.mean(orient_errors)):.1f}°",
    )
    ax2.axvline(
        np.degrees(np.median(orient_errors)),
        color="green",
        linestyle="--",
        label=f"Median: {np.degrees(np.median(orient_errors)):.1f}°",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Show some example predictions
    print("\n" + "=" * 60)
    print("SHOWING EXAMPLE PREDICTIONS")
    print("=" * 60)

    # Show some random examples
    np.random.seed(42)  # For reproducible examples
    num_examples = 5
    random_indices = np.random.choice(len(pos_errors), num_examples, replace=False)

    for i, idx in enumerate(random_indices):
        print(f"\nRandom example {i + 1} (Sample {idx}):")

        # Get prediction and true values
        pred_pos, pred_orient = decode_model_output(predictions[idx])
        true_pos = positions[idx]
        true_orient = orientations[idx]

        print(f"  True position: ({true_pos[0]:.1f}, {true_pos[1]:.1f}) mm")
        print(f"  Pred position: ({pred_pos[0]:.1f}, {pred_pos[1]:.1f}) mm")
        print(f"  True orientation: {np.degrees(true_orient):.1f}°")
        print(f"  Pred orientation: {np.degrees(pred_orient):.1f}°")
        print(f"  Position error: {pos_errors[idx]:.1f}mm")
        print(f"  Orientation error: {np.degrees(orient_errors[idx]):.1f}°")

        # Plot this example
        _ = plot_evaluation_result(
            lidars[idx],
            true_pos,
            true_orient,
            pred_pos,
            pred_orient,
            pos_errors[idx],
            orient_errors[idx],
            idx,
        )
        plt.show()

        # Ask user if they want to continue
        response = input("\nPress Enter to continue to next example, or 'q' to quit: ")
        if response.lower() == "q":
            break

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
