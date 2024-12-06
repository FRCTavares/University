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

    return time, a_data, w_data

def plot_sensor_data(time, a_data, w_data):
    """
    Plot accelerometer and gyroscope data for all axes in the same figure.

    Args:
        time (array): Time data.
        a_data (array): Accelerometer data (Nx3).
        w_data (array): Gyroscope data (Nx3).
    """
    plt.figure()

    # Plot accelerometer data
    plt.subplot(2, 1, 1)
    plt.plot(time, a_data[:, 0], 'r', label='X-axis')
    plt.plot(time, a_data[:, 1], 'g', label='Y-axis')
    plt.plot(time, a_data[:, 2], 'b', label='Z-axis')
    plt.title('Accelerometer Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (mili-g)')
    plt.legend()
    plt.grid(True)

    # Plot gyroscope data
    plt.subplot(2, 1, 2)
    plt.plot(time, w_data[:, 0], 'r', label='X-axis')
    plt.plot(time, w_data[:, 1], 'g', label='Y-axis')
    plt.plot(time, w_data[:, 2], 'b', label='Z-axis')
    plt.title('Gyroscope Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (degrees/s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to load and plot sensor data.
    """
    filename = 'LAB1_6.txt'  # Replace with your data file path
    time, a_data, w_data = load_sensor_data(filename)
    plot_sensor_data(time, a_data, w_data)

if __name__ == "__main__":
    main()
