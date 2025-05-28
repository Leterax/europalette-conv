import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load and analyze the dataset
with open("dataset.pkl", "rb") as f:
    data_list = pickle.load(f)

print(f"Dataset size: {len(data_list)}")

# Extract data
positions = np.stack([d["distance"] for d in data_list], axis=0)
orientations = np.array([d["orientation"] for d in data_list])
lidars = np.stack([d["lidar_data"] for d in data_list], axis=0)

print("\nGround Truth Data Analysis:")
print(
    f"X position: mean={np.mean(positions[:, 0]):.1f}, std={np.std(positions[:, 0]):.1f}, range=[{np.min(positions[:, 0]):.1f}, {np.max(positions[:, 0]):.1f}]"
)
print(
    f"Y position: mean={np.mean(positions[:, 1]):.1f}, std={np.std(positions[:, 1]):.1f}, range=[{np.min(positions[:, 1]):.1f}, {np.max(positions[:, 1]):.1f}]"
)
print(
    f"Orientation: mean={np.degrees(np.mean(orientations)):.1f}°, std={np.degrees(np.std(orientations)):.1f}°, range=[{np.degrees(np.min(orientations)):.1f}°, {np.degrees(np.max(orientations)):.1f}°]"
)

# Check for any constant values
print("\nData diversity check:")
print(f"Unique X positions: {len(np.unique(positions[:, 0].astype(int)))}")
print(f"Unique Y positions: {len(np.unique(positions[:, 1].astype(int)))}")
print(
    f"Unique orientations: {len(np.unique((orientations * 180 / np.pi).astype(int)))}"
)

# Plot distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(positions[:, 0], bins=50, alpha=0.7)
axes[0, 0].set_title("X Position Distribution")
axes[0, 0].set_xlabel("X (mm)")

axes[0, 1].hist(positions[:, 1], bins=50, alpha=0.7)
axes[0, 1].set_title("Y Position Distribution")
axes[0, 1].set_xlabel("Y (mm)")

axes[1, 0].hist(np.degrees(orientations), bins=50, alpha=0.7)
axes[1, 0].set_title("Orientation Distribution")
axes[1, 0].set_xlabel("Orientation (degrees)")

# Check lidar data diversity
valid_readings = []
for lidar in lidars:
    valid_readings.extend(lidar[lidar > 0])

axes[1, 1].hist(valid_readings, bins=50, alpha=0.7)
axes[1, 1].set_title("LiDAR Readings Distribution")
axes[1, 1].set_xlabel("Distance (mm)")

plt.tight_layout()
plt.show()

# Show some example samples
print("\nFirst 10 samples:")
for i in range(min(10, len(data_list))):
    pos = positions[i]
    orient = orientations[i]
    print(
        f"Sample {i}: pos=({pos[0]:.1f}, {pos[1]:.1f}), orient={np.degrees(orient):.1f}°"
    )
