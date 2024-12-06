import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import time
import math
# Define the function to provide the rotation matrix for a given frame
def rotation_matrix_func_original(frame_index):
    return Rotation_Matrix_original[frame_index]
def rotation_matrix_func_extra_cleaned(frame_index):
    return Rotation_Matrix_extra_cleaned[frame_index]

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

def load_and_preprocess_data(filename, scaling_factor):
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
    # Step 1: Define the column index (3 for the fourth column)
    col_index = 2

    # Step 2: Calculate offset and scale
    offset = data[1, col_index]  # First value (row 1, col 3)
    scale_factor = 1000 / (data[187, col_index] - offset)  # Scale to make last value 1000

    # Step 3: Calibrate the column
    data[:, col_index] = (data[:, col_index] - offset) * scale_factor

    time = data[:, 0] / 1e6  # Convert time to seconds
    a_data = data[:, 1:4] * scaling_factor  # Convert accelerometer data to m/s²
    w_data = data[:, 4:7]  # Gyroscope data in rad/s

    return time, a_data, w_data


def compute_velocity_and_position(a_data, time):
    """
    Compute velocity and position by integrating accelerometer data.

    Args:
        a_data (ndarray): Accelerometer data (m/s²).
        time (ndarray): Time array (s).

    Returns:
        tuple: velocity (ndarray), position (ndarray)
    """
    dt = np.diff(time, prepend=time[0])  # Compute time differences
    velocity = np.cumsum(a_data * dt[:, np.newaxis], axis=0)  # Integrate to get velocity
    position = np.cumsum(velocity * dt[:, np.newaxis], axis=0)  # Integrate to get position
    return velocity, position
    
def compute_velocity_and_position_with_rotation(a_data, time, rotation_matrices):
    """
    Compute velocity and position considering rotational transformations.

    Args:
        a_data (ndarray): Accelerometer data (m/s²).
        time (ndarray): Time array (s).
        rotation_matrices (ndarray): Array of rotation matrices.

    Returns:
        tuple: velocity (ndarray), position (ndarray)
    """
    dt = np.diff(time, prepend=time[0])  # Compute time differences
    num_samples = len(a_data)
    
    # Initialize velocity and position arrays
    velocity = np.zeros((num_samples, 3))
    position = np.zeros((num_samples, 3))
    
    for t in range(1, num_samples):
        # Integrate acceleration to update velocity
        velocity[t] = velocity[t-1] + a_data[t] * dt[t]
        
        # Rotate velocity to the global frame and integrate to update position
        global_velocity = rotation_matrices[t] @ velocity[t]
        position[t] = position[t-1] + global_velocity * dt[t]
    
    return velocity, position

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

    print(beta[187])
    # Calibrate angles multipling by 9/8
    alpha=alpha*90/alpha[65]
    beta=beta*90/beta[120]

    return alpha, beta, gamma


def plot_3d_trajectories(positions, labels, colors, title):
    """
    Plot 3D trajectories for multiple datasets.

    Args:
        positions (list): List of position arrays.
        labels (list): List of labels for the datasets.
        colors (list): List of colors for the trajectories.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d')

    for pos, label, color in zip(positions, labels, colors):
        ax.plot3D(pos[:, 0], pos[:, 1], pos[:, 2], label=label, color=color)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_3d_rotations(global_vectors, labels, colors, title):
    

    # Plot the trajectory of the rotated vector
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Obter um mapa de cores (degradê)
    cmap = cm.get_cmap("viridis")  # Escolha um mapa de cores
    colors = cmap(np.linspace(0, 1, len(global_vectors)))  # Gerar as cores
    

    # Plota o caminho com degradê
    for i in range(len(global_vectors) - 1):
        ax.plot(
            [global_vectors[i, 0], global_vectors[i + 1, 0]],
            [global_vectors[i, 1], global_vectors[i + 1, 1]],
            [global_vectors[i, 2], global_vectors[i + 1, 2]],
            color=colors[i],
        )
    # Mark start and end points
    ax.scatter(global_vectors[0, 0], global_vectors[0, 1], global_vectors[0, 2], color='green', s=100, label='Start')
    ax.scatter(global_vectors[-1, 0], global_vectors[-1, 1], global_vectors[-1, 2], color='red', s=100, label='End')
    # Rótulos e título
    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.set_zlabel('Eixo Z')
    ax.set_title('Caminho com Degradê')
    ax.legend()
    '''# Plot the path of the rotated vector
    ax.plot(global_vectors[:, 0], global_vectors[:, 1], global_vectors[:, 2], label='Rotated Vector Path', color='blue')
    ax.scatter([0], [0], [0], color='red', label='Origin', s=50)  # Mark the origin

    # Labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Changes in Rotation Over Time')
    ax.legend()'''

    # Ensure equal aspect ratio
    max_range = np.array([
        global_vectors[:, 0].max() - global_vectors[:, 0].min(),
        global_vectors[:, 1].max() - global_vectors[:, 1].min(),
        global_vectors[:, 2].max() - global_vectors[:, 2].min(),
    ]).max() / 2.0

    mid_x = (global_vectors[:, 0].max() + global_vectors[:, 0].min()) * 0.5
    mid_y = (global_vectors[:, 1].max() + global_vectors[:, 1].min()) * 0.5
    mid_z = (global_vectors[:, 2].max() + global_vectors[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # Show plot
    plt.show()


def visualize_rotation(rotation_matrix_func, num_frames):
    """
    Visualiza um referencial cartesiano rotacionando com base em uma matriz de rotação.

    Args:
        rotation_matrix_func: Função que retorna a matriz de rotação para um dado frame (argumento: índice do frame).
        num_frames: Número de frames na animação.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button

    # Referencial inicial (identidade)
    origin = np.array([0, 0, 0])
    reference_frame = np.eye(3)  # [i, j, k]

    # Configuração do gráfico
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Configuração inicial do referencial
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
    ax.set_title("TItle mudar")

    # Função de atualização para animação
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

        # Update linewidths for fading effect
        for i, q in enumerate(quivers):
            q.set_linewidth(1.5 + 0.2 * i)

    # Criar animação
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

    # Função para pausar e retomar a animação
    is_paused = [False]

    def toggle_pause(event):
        if is_paused[0]:
            ani.event_source.start()
        else:
            ani.event_source.stop()
        is_paused[0] = not is_paused[0]

    # Adicionar botão de pausa
    pause_ax = plt.axes([0.7, 0.01, 0.1, 0.075])
    pause_button = Button(pause_ax, 'Pause')
    pause_button.on_clicked(toggle_pause)

    plt.show()
    
def animate_trajectory(positions, rotations, interval=50):
    """
    Animate a 3D trajectory with the robot's local axes shown and their size scaled to 10% of the trajectory's range.
    
    Args:
        positions (np.ndarray): Array of shape (n, 3) representing the trajectory.
        rotations (np.ndarray): Array of shape (n, 3, 3) representing rotation matrices for each frame.
        interval (int): Time in milliseconds between frames.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize plot elements
    trajectory, = ax.plot([], [], [], lw=2, label="Trajectory")
    point, = ax.plot([], [], [], 'ro', label="Current Position")
    x_axis_line, = ax.plot([], [], [], 'r-', label="X-axis")
    y_axis_line, = ax.plot([], [], [], 'g-', label="Y-axis")
    z_axis_line, = ax.plot([], [], [], 'b-', label="Z-axis")
    
    # Determine the limits for equal scaling
    max_range = np.array([
        np.max(positions[:, 0]) - np.min(positions[:, 0]),
        np.max(positions[:, 1]) - np.min(positions[:, 1]),
        np.max(positions[:, 2]) - np.min(positions[:, 2]),
    ]).max() / 2.0

    mid_x = (np.max(positions[:, 0]) + np.min(positions[:, 0])) / 2.0
    mid_y = (np.max(positions[:, 1]) + np.min(positions[:, 1])) / 2.0
    mid_z = (np.max(positions[:, 2]) + np.min(positions[:, 2])) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set plot labels and title
    ax.set_title("3D Trajectory with Scaled Robot Axes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    def update(frame):
        # Update trajectory and point
        trajectory.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
        trajectory.set_3d_properties(positions[:frame+1, 2])
        point.set_data([positions[frame, 0]], [positions[frame, 1]])
        point.set_3d_properties([positions[frame, 2]])
        
        # Calculate the scaling factor (10% of the trajectory range)
        scale = max_range * 0.1
        
        # Update robot's axes based on the current rotation
        R = rotations[frame]  # Current rotation matrix
        origin = positions[frame]  # Current position
        
        # X-axis (red)
        x_end = origin + R[:, 0] * scale  # Apply rotation to the scaled vector
        x_axis_line.set_data([origin[0], x_end[0]], [origin[1], x_end[1]])
        x_axis_line.set_3d_properties([origin[2], x_end[2]])
        
        # Y-axis (green)
        y_end = origin + R[:, 1] * scale
        y_axis_line.set_data([origin[0], y_end[0]], [origin[1], y_end[1]])
        y_axis_line.set_3d_properties([origin[2], y_end[2]])
        
        # Z-axis (blue)
        z_end = origin + R[:, 2] * scale
        z_axis_line.set_data([origin[0], z_end[0]], [origin[1], z_end[1]])
        z_axis_line.set_3d_properties([origin[2], z_end[2]])
        
        return trajectory, point, x_axis_line, y_axis_line, z_axis_line

    anim = FuncAnimation(fig, update, frames=len(positions), interval=int(interval), blit=False)
    plt.show()

def plot_euler_angles(time_sets, euler_angles, labels, colors, title_prefix):
    """
    Plot Euler angles (Roll, Pitch, Yaw) for multiple datasets.

    Args:
        time_sets (list): List of time arrays for each dataset.
        euler_angles (list): List of Euler angle tuples (alpha, beta, gamma).
        labels (list): List of labels for the datasets.
        colors (list): List of colors for the plots.
        title_prefix (str): Prefix for the plot titles.
    """
    plt.figure(figsize=(10, 6))

    for i, (angle_name, idx) in enumerate(zip(["Roll (α)", "Pitch (β)", "Yaw (γ)"], range(3))):
        plt.subplot(3, 1, i + 1)
        for time, angles, label, color in zip(time_sets, euler_angles, labels, colors):
            plt.plot(time, angles[idx], label=f'{angle_name} - {label}', color=color)
        plt.title(f'{angle_name} - {title_prefix}')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.legend()

    plt.tight_layout()
    plt.show()

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

def compute_gravity_effects(rotation_matrices, data, gravity=9.81):
    """
    Coy_effects: Nx3 array of.
    """
    # Define gravity in the global frame
    gravity_global = np.array([0, 0, -gravity])
    
    gravity_effects = []
    '''
    for R in rotation_matrices:
        # Transform global gravity vector to the sensor frame
        gravity_sensor = R.T @ gravity_global
        gravity_effects.append(gravity_sensor)
    data_without_g=data*10**3+np.array(gravity_effects)
    '''

    for R , data1 in zip(rotation_matrices,data):
        # Transform global gravity vector to the sensor frame
        # Transform global gravity vector to the sensor frame
        gravity_sensor = R.T @ gravity_global
        gravity_effects.append(data1 * 1e3 + gravity_sensor)

    return np.array(gravity_effects)

# Constants
SCALING_FACTOR = 9.81 / 1e6  # Convert accelerometer data from ug to m/s²


# Load and preprocess data
time_original, a_data_original, w_data_original = load_and_preprocess_data(
    'LAB1_6.txt', SCALING_FACTOR)

time_extra_cleaned, a_data_extra_cleaned, w_data_extra_cleaned = load_and_preprocess_data(
    'LAB1_6_denoised_median.txt', SCALING_FACTOR)

# Compute Euler angles
euler_angles_original = compute_euler_angles(w_data_original, time_original)
euler_angles_extra_cleaned = compute_euler_angles(w_data_extra_cleaned, time_extra_cleaned)



# Compute global vectors
global_vectors_original,Rotation_Matrix_original = calc_rotation_matrix(euler_angles_original)
global_vectors_extra_cleaned,Rotation_Matrix_extra_cleaned = calc_rotation_matrix(euler_angles_extra_cleaned)



# Visualization of rotation using the updated rotation_matrix_func
visualize_rotation(rotation_matrix_func_original, num_frames=len(Rotation_Matrix_original))
visualize_rotation(rotation_matrix_func_extra_cleaned, num_frames=len(Rotation_Matrix_extra_cleaned))

'''
#taking the g out
a_data_original_without_g=compute_gravity_effects(Rotation_Matrix_extra_cleaned,a_data_original)
a_data_extra_without_g=compute_gravity_effects(Rotation_Matrix_extra_cleaned,a_data_extra_cleaned)

# Compute velocity and position
_, position_original = compute_velocity_and_position(a_data_original_without_g, time_original)
_, position_extra_cleaned = compute_velocity_and_position(a_data_extra_without_g, time_extra_cleaned)

# Compute velocity and position with rotation
_, position_original_with_rotation = compute_velocity_and_position_with_rotation(a_data_original_without_g, time_original,Rotation_Matrix_extra_cleaned)
_, position_extra_cleaned_with_rotation = compute_velocity_and_position_with_rotation(a_data_extra_without_g, time_extra_cleaned,Rotation_Matrix_extra_cleaned)

# Plot 3D trajectories
plot_3d_trajectories(
    [position_original, position_extra_cleaned],
    labels=["Original Data", "Extra Cleaned Data"],
    colors=['green', 'blue', 'red'],
    title="Comparison of Reconstructed Trajectories without acounting for rotation"
)
plot_3d_trajectories(
    [position_original_with_rotation, position_extra_cleaned_with_rotation],
    labels=["Original Data", "Extra Cleaned Data"],
    colors=['green', 'blue', 'red'],
    title="Comparison of Reconstructed Trajectories in global referencial"
)

animate_trajectory(position_extra_cleaned_with_rotation,Rotation_Matrix_extra_cleaned)
# Plot Euler angles
plot_euler_angles(
    [time_original, time_extra_cleaned],
    [euler_angles_original, euler_angles_extra_cleaned],
    labels=["Original Data", "Cleaned Data", "Extra Cleaned Data"],
    colors=['green', 'blue', 'red'],
    title_prefix="Euler Angle"
)# Plot acceleration comparison
plt.figure(figsize=(12, 8))

# Compare X-axis
plt.subplot(3, 1, 1)
plt.plot(time_original, a_data_original[:, 0]*10**3, label='Original', color='blue')
plt.plot(time_original, a_data_extra_without_g[:, 0], label='denoized Without Gravity', color='red')
plt.title('Acceleration Comparison (X-Axis)')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()

# Compare Y-axis
plt.subplot(3, 1, 2)
plt.plot(time_original, a_data_original[:, 1]*10**3, label='Original', color='blue')
plt.plot(time_original, a_data_extra_without_g[:, 1], label='denoized Without Gravity', color='red')
plt.title('Acceleration Comparison (Y-Axis)')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()

# Compare Z-axis
plt.subplot(3, 1, 3)
plt.plot(time_original, a_data_original[:, 2]*10**3, label='Original', color='blue')
plt.plot(time_original, a_data_extra_without_g[:, 2], label='denoized Without Gravity', color='red')
plt.title('Acceleration Comparison (Z-Axis)')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()

plt.tight_layout()
plt.show()'''