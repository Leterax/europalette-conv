from tqdm import tqdm
import numpy as np
from raycast import raycast_axis_aligned_squares_batch_ultra
import matplotlib.pyplot as plt
import time

# Configuration constants
LIDAR_MAX_RANGE_MM = 30000  # 30m maximum lidar range
PALLET_MIN_DISTANCE_MM = 500  # 0.5m minimum distance to pallets
PALLET_MAX_DISTANCE_MM = 1000  # 10m maximum distance to pallets
VIEW_ANGLE_RANDOMNESS_RAD = 140 * np.pi / 180.0  # ±140° × π/180
FIELD_OF_VIEW_DEGREES = 270  # Field of view for raycast
NUM_RAYS = 200  # Number of rays to cast
RAY_SAMPLE_RATE = 1  # Show every Nth ray in visualization

# Define the squares (europalettes)
vertices = np.array(
    [
        [0.0, 0.0],
        [145.0, 0.0],
        [145.0, 100.0],
        [0.0, 100.0],
        [527.5, 0.0],
        [672.5, 0.0],
        [672.5, 100.0],
        [527.5, 100.0],
        [1055.0, 0.0],
        [1200.0, 0.0],
        [1200.0, 100.0],
        [1055.0, 100.0],
        [0.0, 327.5],
        [145.0, 327.5],
        [145.0, 472.5],
        [0.0, 472.5],
        [527.5, 327.5],
        [672.5, 327.5],
        [672.5, 472.5],
        [527.5, 472.5],
        [1055.0, 327.5],
        [1200.0, 327.5],
        [1200.0, 472.5],
        [1055.0, 472.5],
        [0.0, 700.0],
        [145.0, 700.0],
        [145.0, 800.0],
        [0.0, 800.0],
        [527.5, 700.0],
        [672.5, 700.0],
        [672.5, 800.0],
        [527.5, 800.0],
        [1055.0, 700.0],
        [1200.0, 700.0],
        [1200.0, 800.0],
        [1055.0, 800.0],
    ]
)

squares = vertices.reshape(9, 4, 2)


def get_workspace_bounds(squares):
    """Calculate the bounding box of the entire workspace."""
    all_points = squares.reshape(-1, 2)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    return min_x, min_y, max_x, max_y


def generate_random_origin_and_view(workspace_bounds, margin=None):
    """Generate a random origin outside the workspace and a view direction."""
    min_x, min_y, max_x, max_y = workspace_bounds
    workspace_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])

    # Random margin between min and max pallet distance
    if margin is None:
        margin = np.random.uniform(PALLET_MIN_DISTANCE_MM, PALLET_MAX_DISTANCE_MM)

    # Generate origin outside workspace with margin
    side = np.random.choice(["left", "right", "top", "bottom"])

    if side == "left":
        origin = np.array(
            [min_x - margin, np.random.uniform(min_y - margin, max_y + margin)]
        )
    elif side == "right":
        origin = np.array(
            [max_x + margin, np.random.uniform(min_y - margin, max_y + margin)]
        )
    elif side == "top":
        origin = np.array(
            [np.random.uniform(min_x - margin, max_x + margin), max_y + margin]
        )
    else:  # bottom
        origin = np.array(
            [np.random.uniform(min_x - margin, max_x + margin), min_y - margin]
        )

    # Calculate base angle towards workspace center
    to_center = workspace_center - origin
    base_angle = np.arctan2(to_center[1], to_center[0])

    # Add some randomness to the view direction
    view_angle = base_angle + np.random.uniform(
        -VIEW_ANGLE_RANDOMNESS_RAD, VIEW_ANGLE_RANDOMNESS_RAD
    )

    return origin, view_angle


def cast_rays(squares, origin, view_angle, fov_degrees=None, num_rays=None):
    """Cast rays from origin with given view angle and field of view."""
    if fov_degrees is None:
        fov_degrees = FIELD_OF_VIEW_DEGREES
    if num_rays is None:
        num_rays = NUM_RAYS

    fov_radians = np.radians(fov_degrees)
    angles = np.linspace(
        view_angle - fov_radians / 2, view_angle + fov_radians / 2, num_rays
    )
    distances = raycast_axis_aligned_squares_batch_ultra(squares, origin, angles)
    return angles, distances


def transform_point(point, origin, view_angle):
    """Transform a point to the coordinate frame where origin is at (0,0) looking towards +X."""
    # Translate so origin is at (0,0)
    translated = point - origin

    # Rotate so view direction aligns with +X axis
    cos_rot = np.cos(-view_angle)
    sin_rot = np.sin(-view_angle)
    rotation_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])

    return rotation_matrix @ translated


def transform_points(points, origin, view_angle):
    """Transform multiple points to the new coordinate frame."""
    translated = points - origin
    cos_rot = np.cos(-view_angle)
    sin_rot = np.sin(-view_angle)
    rotation_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])

    return (rotation_matrix @ translated.T).T


def plot_raycast_result(
    squares, origin, view_angle, angles, distances, ground_truth=None, title_suffix=""
):
    """Plot the raycast visualization in transformed coordinate frame."""
    plt.figure(figsize=(12, 8))

    # Plot the squares as filled rectangles in transformed coordinates
    for i, square in enumerate(squares):
        transformed_square = transform_points(square, origin, view_angle)
        xs, ys = transformed_square[:, 0], transformed_square[:, 1]
        # Close the polygon by adding the first point at the end
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])

        # Highlight center block differently
        if i == 4:  # Center block
            plt.fill(
                xs, ys, alpha=0.5, color="orange", edgecolor="darkorange", linewidth=2
            )
        else:
            plt.fill(xs, ys, alpha=0.3, color="blue", edgecolor="darkblue")

    # Plot the origin (now at 0,0)
    plt.scatter(
        0, 0, c="red", s=100, marker="o", label="Origin", zorder=5, edgecolor="darkred"
    )

    # Plot ground truth pallet center if provided
    if ground_truth is not None:
        position_relative, orientation_relative, distance = ground_truth
        plt.scatter(
            position_relative[0],
            position_relative[1],
            c="purple",
            s=150,
            marker="x",
            label="Pallet Center (GT)",
            zorder=6,
            linewidth=3,
        )

        # Draw orientation arrow
        arrow_length = 200  # mm
        arrow_end = position_relative + arrow_length * np.array(
            [np.cos(orientation_relative), np.sin(orientation_relative)]
        )
        plt.arrow(
            position_relative[0],
            position_relative[1],
            arrow_end[0] - position_relative[0],
            arrow_end[1] - position_relative[1],
            head_width=50,
            head_length=30,
            fc="purple",
            ec="purple",
            label="Pallet Orientation",
            zorder=6,
        )

    # Draw sample rays (every Nth ray for cleaner visualization)
    sample_indices = np.arange(0, len(angles), RAY_SAMPLE_RATE)
    for i in sample_indices:
        angle = angles[i]
        distance = distances[i]

        # In the transformed coordinate frame, ray directions are relative to the view angle
        transformed_angle = angle - view_angle
        ray_dir = np.array([np.cos(transformed_angle), np.sin(transformed_angle)])

        if distance != -1:
            # Ray hits something - draw to hit point
            hit_point = distance * ray_dir
            plt.plot(
                [0, hit_point[0]], [0, hit_point[1]], "g-", alpha=0.6, linewidth=0.5
            )
        else:
            # Ray doesn't hit - draw a long ray in that direction
            max_distance = LIDAR_MAX_RANGE_MM  # Use lidar max range
            end_point = max_distance * ray_dir
            plt.plot(
                [0, end_point[0]], [0, end_point[1]], "r-", alpha=0.3, linewidth=0.5
            )

    # Calculate hit points for rays that intersect
    valid_indices = distances != -1
    if np.any(valid_indices):
        valid_distances = distances[valid_indices]
        valid_angles = angles[valid_indices]

        # In transformed coordinate frame, calculate hit points relative to view angle
        transformed_angles = valid_angles - view_angle
        ray_directions = np.column_stack(
            [np.cos(transformed_angles), np.sin(transformed_angles)]
        )
        hit_points = valid_distances[:, np.newaxis] * ray_directions

        # Plot hit points colored by distance
        scatter = plt.scatter(
            hit_points[:, 0],
            hit_points[:, 1],
            c=valid_distances,
            cmap="viridis",
            s=2,
            label="Hit points",
            alpha=0.8,
        )
        plt.colorbar(scatter, label="Distance", shrink=0.8)

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("X coordinate (forward direction)")
    plt.ylabel("Y coordinate (left/right)")

    # Calculate statistics for title
    num_hits = np.sum(valid_indices)
    hit_rate = num_hits / len(distances) * 100

    title = f"Raycast Visualization{title_suffix}\n"
    title += f"Origin: ({origin[0]:.0f}, {origin[1]:.0f}), View: {np.degrees(view_angle):.0f}°, Hits: {num_hits}/{len(distances)} ({hit_rate:.1f}%)"

    if ground_truth is not None:
        position_relative, orientation_relative, distance = ground_truth
        title += f"\nPallet: ({position_relative[0]:.0f}, {position_relative[1]:.0f}), "
        title += (
            f"Dist: {distance:.0f}mm, Orient: {np.degrees(orientation_relative):.0f}°"
        )

    plt.title(title)
    plt.legend()
    plt.tight_layout()


def print_raycast_info(origin, view_angle, angles, distances):
    """Print information about the raycast setup and results."""
    workspace_bounds = get_workspace_bounds(squares)
    workspace_center = np.array(
        [
            (workspace_bounds[0] + workspace_bounds[2]) / 2,
            (workspace_bounds[1] + workspace_bounds[3]) / 2,
        ]
    )

    print(f"Origin: ({origin[0]:.1f}, {origin[1]:.1f})")
    print(f"Workspace center: ({workspace_center[0]:.1f}, {workspace_center[1]:.1f})")
    print(f"View angle: {view_angle:.3f} rad ({np.degrees(view_angle):.1f}°)")
    print(f"Ray range: {np.degrees(angles[0]):.1f}° to {np.degrees(angles[-1]):.1f}°")

    valid_hits = np.sum(distances != -1)
    hit_rate = valid_hits / len(distances) * 100
    print(f"Hits: {valid_hits}/{len(distances)} ({hit_rate:.1f}%)")

    if valid_hits > 0:
        valid_distances = distances[distances != -1]
        print(
            f"Distance range: {np.min(valid_distances):.1f} to {np.max(valid_distances):.1f}"
        )
        print(f"Mean distance: {np.mean(valid_distances):.1f}")
    print("-" * 50)


def get_pallet_center_and_orientation(squares):
    """
    Calculate the center position and orientation of the europalette.

    The europalette is arranged as a 3x3 grid of squares:
    [0] [1] [2]
    [3] [4] [5]
    [6] [7] [8]

    The center block is square[4].
    """
    # Center block is index 4 (middle of 3x3 grid)
    center_block = squares[4]

    # Calculate center of the center block
    center_position = np.mean(center_block, axis=0)

    # Calculate orientation based on the pallet's structure
    # Use the vector from center of left column to center of right column
    left_column_center = np.mean(
        [
            np.mean(squares[0], axis=0),
            np.mean(squares[3], axis=0),
            np.mean(squares[6], axis=0),
        ],
        axis=0,
    )
    right_column_center = np.mean(
        [
            np.mean(squares[2], axis=0),
            np.mean(squares[5], axis=0),
            np.mean(squares[8], axis=0),
        ],
        axis=0,
    )

    # Orientation vector (from left to right)
    orientation_vector = right_column_center - left_column_center
    orientation_angle = np.arctan2(orientation_vector[1], orientation_vector[0])

    return center_position, orientation_angle


def calculate_pallet_ground_truth(squares, origin, view_angle):
    """
    Calculate the ground truth position and orientation of the pallet
    relative to the sensor origin and view direction.

    Returns:
        position_relative: (x, y) position of pallet center in transformed coordinates
        orientation_relative: orientation of pallet relative to sensor view direction
        distance: distance from origin to pallet center
    """
    # Get pallet center and orientation in world coordinates
    pallet_center, pallet_orientation = get_pallet_center_and_orientation(squares)

    # Transform pallet center to sensor coordinate frame
    position_relative = transform_point(pallet_center, origin, view_angle)

    # Calculate relative orientation (pallet orientation - sensor view angle)
    orientation_relative = pallet_orientation - view_angle

    # Normalize angle to [-π, π]
    orientation_relative = np.arctan2(
        np.sin(orientation_relative), np.cos(orientation_relative)
    )

    # palets are symetric so +-180 deg makes no difference
    # So if orientation < 0 add pi
    if orientation_relative < 0:
        orientation_relative += np.pi

    # Calculate distance from origin to pallet center
    distance = np.linalg.norm(position_relative)

    return position_relative, orientation_relative, distance


def main():
    """Generate and visualize several random raycast examples."""
    workspace_bounds = get_workspace_bounds(squares)
    print(
        f"Workspace bounds: X=({workspace_bounds[0]}, {workspace_bounds[2]}), "
        f"Y=({workspace_bounds[1]}, {workspace_bounds[3]})"
    )
    print("=" * 50)

    # Generate 4 random examples
    np.random.seed(42)  # For reproducible results
    dataset = []
    start = time.time()
    for i in tqdm(range(65536 * 2)):
        # print(f"Example {i + 1}:")

        # Generate random origin and view direction
        origin, view_angle = generate_random_origin_and_view(workspace_bounds)

        # Calculate ground truth position and orientation of the pallet
        ground_truth = calculate_pallet_ground_truth(squares, origin, view_angle)
        position_relative, orientation_relative, distance = ground_truth

        # print(f"Ground Truth Pallet Information:")
        # print(
        #     f"  Position (x, y): ({position_relative[0]:.1f}, {position_relative[1]:.1f}) mm"
        # )
        # print(f"  Distance from origin: {distance:.1f} mm")
        # print(
        #     f"  Orientation relative to view: {np.degrees(orientation_relative):.1f}°"
        # )

        # Cast rays
        angles, distances = cast_rays(squares, origin, view_angle)

        dataset.append(
            {
                "distance": position_relative,
                "orientation": orientation_relative,
                "lidar_data": distances,
            }
        )

        # # Print info
        # print_raycast_info(origin, view_angle, angles, distances)

        # # Plot results
        if i > 3:
            continue

        plot_raycast_result(
            squares,
            origin,
            view_angle,
            angles,
            distances,
            ground_truth,
            title_suffix=f" - Example {i + 1}",
        )

        # plot the raw data as a line plot
        plt.figure(figsize=(10, 5))
        plt.plot(distances)
        plt.title("Lidar Data")
        plt.xlabel("Ray Index")
        plt.ylabel("Distance (mm)")
        plt.show()
        # print("=" * 50)
    end = time.time()

    print(f"Time taken: {end - start} seconds")
    print(f"RPS: {1000 / (end - start)}")

    # save dataset to file
    import pickle

    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
