
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

def rotation_matrix_func(frame_index):
    return Final_rotation_matrix[frame_index]

def rotation_matrix_x(alpha):
    return np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

def rotation_matrix_y(beta):
    return np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

def rotation_matrix_z(gamma):
    return np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])


def visualize_rotation(rotation_matrix_func, num_frames):
    """
    Visualize a rotating Cartesian reference frame based on a rotation matrix.

    Args:
        rotation_matrix_func: Function that returns the rotation matrix for a given frame (argument: frame index).
        num_frames: Number of frames in the animation.
    """

    # Initial reference frame (identity matrix)
    origin = np.array([0, 0, 0])
    reference_frame = np.eye(3)  # [i, j, k]

    # Graph configuration
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Initial reference frame configuration
    quivers = []
    for _ in range(3):
        quiver = ax.quiver(
            origin[0], origin[1], origin[2],
            reference_frame[0, :], reference_frame[1, :], reference_frame[2, :],
            color=['red', 'green', 'blue'], linewidth=2, arrow_length_ratio=0, length=1.5
        )
        quivers.append(quiver)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend for axis colors
    red_line = plt.Line2D([0], [0], color='red', lw=2, label='Roll (alpha)')
    green_line = plt.Line2D([0], [0], color='green', lw=2, label='Pitch (beta)')
    blue_line = plt.Line2D([0], [0], color='blue', lw=2, label='Yaw (gamma)')
    ax.legend(handles=[red_line, green_line, blue_line])

    # Update function for animation
    def update(frame):
        rotation_matrix = rotation_matrix_func(frame)
        rotated_frame = rotation_matrix @ reference_frame

        # Remove the oldest quiver if more than 10
        if len(quivers) > 3:
            quivers.pop(0).remove()

        # Add the new quiver
        quiver = ax.quiver(
            origin[0], origin[1], origin[2],
            rotated_frame[0, :], rotated_frame[1, :], rotated_frame[2, :],
            color=['red', 'green', 'blue'], linewidth=3, arrow_length_ratio=0, length=1.5
        )
        quivers.append(quiver)

        # Update linewidths and alpha for fading effect
        for i, q in enumerate(quivers):
            alpha = (i + 1) / len(quivers)  # Calculate alpha based on position in the list
            q.set_alpha(alpha)
            q.set_linewidth(1.5 + 0.2 * i)

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

    # Function to pause and resume the animation
    is_paused = [False]

    def toggle_pause(event):
        if is_paused[0]:
            ani.event_source.start()
        else:
            ani.event_source.stop()
        is_paused[0] = not is_paused[0]

    # Add pause button
    pause_ax = plt.axes([0.7, 0.01, 0.1, 0.075])
    pause_button = Button(pause_ax, 'Pause')
    pause_button.on_clicked(toggle_pause)

    plt.show()



Final_rotation_matrix=[]
# Start with Identity rotation
rotation_matrix = rotation_matrix_z(np.radians(0)) @ rotation_matrix_y(np.radians(0)) @ rotation_matrix_x(np.radians(0))
Final_rotation_matrix.append(rotation_matrix)

# Z-axis rotations
rotation_matrix = rotation_matrix_z(np.radians(-5))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-15))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-22))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-29))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-36))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-44))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-51))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-58))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-65))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-72.5))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-80))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)

# X-axis and Z-axis combined rotations
rotation_matrix = rotation_matrix_x(np.radians(10)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(15)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(22)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(29)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(36)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(43.5)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(50)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(58)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(65)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(72.5)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(79)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)

# Y-axis, X-axis, and Z-axis combined rotations
rotation_matrix = rotation_matrix_y(np.radians(-5)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-15)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-22)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-29)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-36)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-43.5)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-50)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-58)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-65)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-72.5)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-79)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)

# Z-axis, Y-axis, and X-axis combined rotations
rotation_matrix = rotation_matrix_z(np.radians(5)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(15)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(22)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(29)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(36)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(43.5)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(50)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(58)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(65)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(72.5)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(79)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)
rotation_matrix = rotation_matrix_z(np.radians(87.8)) @ rotation_matrix_y(np.radians(-86.3)) @ rotation_matrix_x(np.radians(86.2)) @ rotation_matrix_z(np.radians(-87))
Final_rotation_matrix.append(rotation_matrix)




visualize_rotation(rotation_matrix_func, num_frames=len(Final_rotation_matrix))