import numpy as np
import matplotlib.pyplot as plt

def load_sensor_data(filename):
    """
    Load sensor data from a file.

    Args:
        filename (str): Path to the data file.

    Returns:
        tuple: time (s), accelerometer data (Nx3), gyroscope data (Nx3)
    """
    data = np.loadtxt(filename)
    time = data[:, 0] / 1e6  # Convert time to seconds
    a_data = data[:, 1:4]    # Accelerometer data (x, y, z)
    w_data = data[:, 4:7]    # Gyroscope data (x, y, z)

    a_data = a_data

    return time, a_data, w_data


def plot_sensor_data(time, a_data, w_data):
    """
    Plot accelerometer and gyroscope data for all axes.

    Args:
        time (ndarray): Time array (s).
        a_data (ndarray): Accelerometer data (Nx3).
        w_data (ndarray): Gyroscope data (Nx3).
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))

    # Plot accelerometer data
    plt.subplot(2, 1, 1)
    plt.plot(time, a_data[:, 0], label='Accel X', color='red')
    plt.plot(time, a_data[:, 1], label='Accel Y', color='green')
    plt.plot(time, a_data[:, 2], label='Accel Z',  color='blue')

    # Add labels, legend, and grid
    plt.title("Accelerometer Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (mili-g)")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Plot gyroscope data
    plt.subplot(2, 1, 2)
    plt.plot(time, w_data[:, 0], label='Gyro X', color='red')
    plt.plot(time, w_data[:, 1], label='Gyro Y', color='green')
    plt.plot(time, w_data[:, 2], label='Gyro Z',  color='blue')

    # Add labels, legend, and grid
    plt.title("Gyroscope Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (degrees/s)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_axis_data(time, a_data, w_data):
    """
    Plot accelerometer and gyroscope data together for each axis in separate subplots.

    Args:
        time (ndarray): Time array (s).
        a_data (ndarray): Accelerometer data (Nx3).
        w_data (ndarray): Gyroscope data (Nx3).
    """
    axes = ['X', 'Y', 'Z']
    colors_accel = ['red', 'green', 'blue']
    colors_gyro = ['orange', 'limegreen', 'skyblue']

    plt.figure(figsize=(12, 8))

    for i, axis in enumerate(axes):
        plt.subplot(3, 1, i + 1)
        plt.plot(time, a_data[:, i], label=f'Accel {axis}', color=colors_accel[i])
        plt.plot(time, w_data[:, i], label=f'Gyro {axis}',  color=colors_gyro[i])
        plt.title(f'{axis}-Axis Sensor Data')
        plt.xlabel("Time (s)")
        plt.ylabel("Sensor Readings")
        plt.legend(loc="upper right")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# Main script
filename = "LAB1_6.txt"

# Load data
time, a_data, w_data = load_sensor_data(filename)

# Plot all sensor data together
plot_sensor_data(time, a_data, w_data)

# Plot data for each axis separately
#plot_axis_data(time, a_data, w_data)
