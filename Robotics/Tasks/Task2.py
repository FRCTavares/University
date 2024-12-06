import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter

# Helper Functions
def load_data(filename):
    """
    Load sensor data from a file.

    Args:
        filename (str): Path to the data file.

    Returns:
        tuple: time (s), accelerometer data, gyroscope data
    """
    data = np.loadtxt(filename)
    time = data[:, 0] / 1e6  # Convert time to seconds
    a_data = data[:, 1:4]    # Accelerometer data (x, y, z)
    w_data = data[:, 4:7]    # Rate-gyro data (x, y, z)
    return time, a_data, w_data


def apply_denoising(data, method='median', **kwargs):
    """
    Apply denoising to the sensor data.

    Args:
        data (ndarray): Sensor data (Nx3).
        method (str): Denoising method ('median', 'savgol', 'aaf').
        kwargs: Additional arguments for the denoising method.

    Returns:
        ndarray: Denoised data.
    """
    if method == 'median':
        kernel_size = kwargs.get('kernel_size', 7)
        return np.array([medfilt(data[:, i], kernel_size) for i in range(data.shape[1])]).T
    elif method == 'savgol':
        window_length = kwargs.get('window_length', 21)
        polyorder = kwargs.get('polyorder', 3)
        return np.array([savgol_filter(data[:, i], window_length, polyorder) for i in range(data.shape[1])]).T
    elif method == 'aaf':  # Simple Agressive Amplitude Filter
        # If the value is bellow the threshold, it is set to 0
        amp = kwargs.get('amp', 10)
        return np.where(np.abs(data) > amp, data, 0)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def plot_combined_data(time, original_a, processed_a, original_w, processed_w, title_prefix):
    """
    Plot all accelerometer and gyroscope data (X, Y, Z) in the same figure.

    Args:
        time (ndarray): Time array (s).
        original_a (ndarray): Original accelerometer data (Nx3).
        processed_a (ndarray): Processed accelerometer data (Nx3).
        original_w (ndarray): Original gyroscope data (Nx3).
        processed_w (ndarray): Processed gyroscope data (Nx3).
        title_prefix (str): Prefix for the plot title.
    """

    plt.figure(figsize=(12, 8))

    # Accelerometer data subplot
    plt.subplot(2, 1, 1)
    colors = ['red', 'green', 'blue']
    axis_labels = ['X', 'Y', 'Z']
    for i in range(3):
        plt.plot(time, original_a[:, i], linestyle='--', color=colors[i], alpha=0.5, label=f'Original {axis_labels[i]}')
        plt.plot(time, processed_a[:, i], color=colors[i], label=f'Processed {axis_labels[i]}')
    plt.title(f"{title_prefix} - Accelerometer Data")
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (mili-g)')
    plt.legend()
    plt.grid(True)

    # Gyroscope data subplot
    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(time, original_w[:, i], linestyle='--', color=colors[i], alpha=0.5, label=f'Original {axis_labels[i]}')
        plt.plot(time, processed_w[:, i], color=colors[i], label=f'Processed {axis_labels[i]}')
    plt.title(f"{title_prefix} - Gyroscope Data")
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (degrees/s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def save_data(filename, time, a_data, w_data):
    """
    Save sensor data to a file.

    Args:
        filename (str): Path to the output data file.
        time (ndarray): Time array (s).
        a_data (ndarray): Accelerometer data (Nx3).
        w_data (ndarray): Gyroscope data (Nx3).
    """
    data = np.hstack((time[:, None] * 1e6, a_data, w_data))  # Convert time back to microseconds
    np.savetxt(filename, data, fmt='%.6f')


# Menu and Execution
def main_menu():
    print("Select the processing method:")
    print("1. Median Filter + Agressive Amplitude Filter")
    print("2. Savitzky-Golay Filter + Agressive Amplitude Filter")
    print("3. Exit")
    choice = input("Enter your choice: ")
    return choice


def main():
    # Load data
    filename = "LAB1_6.txt"
    time, a_data, w_data = load_data(filename)

    while True:
        choice = main_menu()

        if choice == "1":
            # Median Filter
            a_data_denoised = apply_denoising(a_data, method='median', kernel_size=11)
            w_data_denoised = apply_denoising(w_data, method='median', kernel_size=11)
            
            # Further denoising by applying an aggressive amplitude filter
            a_data_denoised = apply_denoising(a_data_denoised, method='aaf', amp=60)
            w_data_denoised = apply_denoising(w_data_denoised, method='aaf', amp=21)

            # Plot results
            plot_combined_data(time, a_data, a_data_denoised, w_data, w_data_denoised, "Median Filter + Aggressive Amplitude Filter")

            # Save new text files
            save_data("LAB1_6_denoised_median.txt", time, a_data_denoised, w_data_denoised)

        elif choice == "2":
            # Savitzky-Golay Filter
            a_data_denoised = apply_denoising(a_data, method='savgol', window_length=11, polyorder=2)
            w_data_denoised = apply_denoising(w_data, method='savgol', window_length=11, polyorder=2)

            # Further denoising by applying an aggressive amplitude filter
            a_data_denoised = apply_denoising(a_data_denoised, method='aaf', amp=60)
            w_data_denoised = apply_denoising(w_data_denoised, method='aaf', amp=21)

            # Plot results
            plot_combined_data(time, a_data, a_data_denoised, w_data, w_data_denoised, "Savitzky-Golay Filter + Aggressive Amplitude Filter")

            # Save new text files
            save_data("LAB1_6_denoised_savgol.txt", time, a_data_denoised, w_data_denoised)

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice! Please try again.")


if __name__ == "__main__":
    main()
