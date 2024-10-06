import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_landmarks(csv_file='data/processed/labeled_hand_landmarks.csv', frame_number=0):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Select the specific frame
    frame_data = df.iloc[frame_number]

    # Extract hand landmarks
    hand_landmarks = frame_data.iloc[:63].values.reshape(
        (21, 3))  # 21 keypoints × 3 coordinates

    # Extract label
    label = frame_data.iloc[-1]

    # Plot the landmarks
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Frame {frame_number} - Label: {label}")

    # Plot each landmark
    for i, (x, y, z) in enumerate(hand_landmarks):
        ax.scatter(x, y, z, label=f'Point {i}')
        ax.text(x, y, z, f'{i}', size=10, zorder=1)

    # Set axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example: Visualize the first frame
    visualize_landmarks(frame_number=0)
