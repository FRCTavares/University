import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button



def load_and_preprocess_data(filename):
    """
    Load sensor data, preprocess accelerometer data to remove gravity bias,
    and return time, accelerometer, and gyroscope data.
    
    Args:
        filename (str): Path to the data file.
        scaling_factor (float): Scaling factor to convert accelerometer data.
        static_samples (int): Number of samples to calculate the gravity bias.

    Returns:
        tuple: time, accelerometer data, gyroscope data
    """
    data = np.loadtxt(filename)

    time = data[:, 0] / 1e6  # Convert time to seconds
    w_data = data[:, 4:7]  # Gyroscope data in degrees/s


    return time, w_data


def compute_euler_angles(w_data, time):
    """
    Compute Euler angles (roll, pitch, yaw) by integrating gyroscope data.

    Args:
        w_data (ndarray): Gyroscope data (rad/s).
        time (ndarray): Time array (s).

    Returns:
        tuple: Roll (alpha), Pitch (beta), Yaw (gamma)
    """
    dt = np.diff(time, prepend=time[0])  # Compute time differences
    alpha = np.cumsum(w_data[:, 0] * dt)  # Roll
    beta = np.cumsum(w_data[:, 1] * dt)   # Pitch
    gamma = np.cumsum(w_data[:, 2] * dt)  # Yaw

    # Calibrate angles multipling by 9/8
    alpha=alpha*90/alpha[65]
    beta=beta*90/beta[120]

    return alpha, beta, gamma


def calc_rotation_matrix(euler_angles):

    # Fixed vector in the body frame
    body_vector = np.array([0, 0, 1])

    # Store the global vectors over time
    global_vectors = []
    Rs=[]
    euler_angles = np.column_stack(euler_angles)
    for alpha, beta, gamma in euler_angles:
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)
        # Compute the rotation matrix
        R = rotation_matrix_x(alpha) @ rotation_matrix_y(beta) @ rotation_matrix_z(gamma) 
        
        # Transform the body vector to the global frame
        global_vector = R @ body_vector
        Rs.append(R)
        global_vectors.append(global_vector)
    # Convert list to numpy array for easier handling
    global_vectors = np.array(global_vectors)
    Rs=np.array(Rs)
    
    return global_vectors,Rs


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
    for _ in range(10):
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
        if len(quivers) > 10:
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



def main():
    # Load and preprocess data
    time_original, w_data_original = load_and_preprocess_data('LAB1_6.txt')

    time_extra_cleaned, w_data_extra_cleaned = load_and_preprocess_data('LAB1_6_denoised_median.txt')

    # Compute Euler angles
    euler_angles_original = compute_euler_angles(w_data_original, time_original)
    euler_angles_extra_cleaned = compute_euler_angles(w_data_extra_cleaned, time_extra_cleaned)

    # Compute global vectors
    global_vectors_original,Rotation_Matrix_original = calc_rotation_matrix(euler_angles_original)
    global_vectors_extra_cleaned,Rotation_Matrix_extra_cleaned = calc_rotation_matrix(euler_angles_extra_cleaned)

    # Define the function to provide the rotation matrix for a given frame
    def rotation_matrix_func_original(frame_index):
        return Rotation_Matrix_original[frame_index]

    def rotation_matrix_func_extra_cleaned(frame_index):
        return Rotation_Matrix_extra_cleaned[frame_index]


    # Visualization of rotation using the updated rotation_matrix_func
    visualize_rotation(rotation_matrix_func_original, num_frames=len(Rotation_Matrix_original))
    visualize_rotation(rotation_matrix_func_extra_cleaned, num_frames=len(Rotation_Matrix_extra_cleaned))



if __name__ == "__main__":
    main()