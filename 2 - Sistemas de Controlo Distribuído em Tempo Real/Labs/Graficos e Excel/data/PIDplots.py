import re
import csv
import matplotlib.pyplot as plt
import numpy as np

# Path to your text file with the data dump
txt_filename = "test.txt"

# Path to your desired CSV output
csv_filename = "output.csv"

def parse_line(label, whole_text):
    """
    Searches for a line in 'whole_text' that starts with e.g. 'Time:'
    and returns the list of floats after that colon.
    """
    pattern = rf"{label}:\s*(.*)"
    match = re.search(pattern, whole_text)
    if not match:
        return []
    # This is the part after "<label>:" 
    floats_part = match.group(1)
    # Split on commas, strip spaces, convert to float
    return [float(x.strip()) for x in floats_part.split(',') if x.strip()]

# Function to calculate settling time (within 2% of final value)
def calculate_settling_time(time, measured, reference, tolerance=0.02):
    # Get the final/target value
    final_value = reference[-1]
    # Calculate the tolerance band
    lower_bound = final_value * (1 - tolerance)
    upper_bound = final_value * (1 + tolerance)
    
    # Find where signal enters and stays within the tolerance band
    for i in range(len(measured)):
        if lower_bound <= measured[i] <= upper_bound:
            # Check if it stays within bounds for the rest of the data
            staying_within = True
            for j in range(i, len(measured)):
                if measured[j] < lower_bound or measured[j] > upper_bound:
                    staying_within = False
                    break
            if staying_within:
                return time[i]
    return None  # If settling never occurs

# Function to calculate percentage overshoot
def calculate_overshoot(measured, reference):
    # Get the final/target value
    final_value = reference[-1]
    # Find the maximum value after the step input
    max_value = np.max(measured)
    # Calculate the percentage overshoot
    if max_value > final_value:
        return ((max_value - final_value) / final_value) * 100
    else:
        return 0  # No overshoot

# 1) Read the entire text file
with open(txt_filename, "r") as f:
    content = f.read()

# 2) Parse each named array from the text
time_data       = np.array(parse_line("Time", content))
measured_data   = np.array(parse_line("MeasuredLux", content))
duty_data       = np.array(parse_line("DutyCycle", content))
setpoint_data   = np.array(parse_line("SetpointLux", content))

# 3) Write them all to a CSV (4 columns)
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    # Write a header row
    writer.writerow(["Time", "MeasuredLux", "DutyCycle", "SetpointLux"])
    
    # Zip all arrays so we can write row by row
    for row in zip(time_data, measured_data, duty_data, setpoint_data):
        writer.writerow(row)

# 4) Prepare data for plotting
# Convert time to seconds and shift to start at zero
time_zero = time_data - time_data[0]
time_seconds = time_data / 1000  # Convert to seconds

# Filter out first 15 seconds and limit to next 30 seconds (15-45s total)
idx = (time_zero >= 8500) & (time_zero <= 23500)  # Between 15 and 45 seconds (in milliseconds)
time_filtered = time_zero[idx] / 1000 - 8.5 # Convert to seconds and re-zero to start at 0
measured_filtered = measured_data[idx]
duty_filtered = duty_data[idx]
reference_filtered = setpoint_data[idx]

# Calculate performance metrics
settling_time = calculate_settling_time(time_filtered, measured_filtered, reference_filtered)
overshoot = calculate_overshoot(measured_filtered, reference_filtered)

# 5) Create plot with dual y-axes
plt.figure(figsize=(10, 8), num="Lux & DutyCycle")

# Left y-axis for Lux values
ax1 = plt.gca()
p1 = ax1.plot(time_filtered, reference_filtered, '#1f77b4', linewidth=1.7, label='Reference Lux')  # Deep blue
p2 = ax1.plot(time_filtered, measured_filtered, '#2ca02c', linewidth=1.7, label='Measured Lux')    # Green
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Lux [lx]')
ax1.set_ylim(0, 25)  # Set Lux limit to 25
ax1.grid(True)

# Right y-axis for Duty Cycle
ax2 = ax1.twinx()
p3 = ax2.plot(time_filtered, duty_filtered, '#ff7f0e', linewidth=1.7, label='Duty Cycle')          # Orange
ax2.set_ylabel('Duty Cycle [0..1]')
ax2.set_ylim(0, 1)  # Set Duty Cycle limit to 1

# Add performance metrics to the plot
if settling_time is None:
    metrics_text = f"Settling Time: Never settles\nOvershoot: {overshoot:.2f}%"
else:
    metrics_text = f"Settling Time: {settling_time:.2f} s\nOvershoot: {overshoot:.2f}%"
plt.figtext(0.15, 0.15, metrics_text, bbox=dict(facecolor='white', alpha=0.8))


# Title and legend
plt.title('Reference Lux, Measured Lux, and Duty Cycle vs. Time')
lines = p1 + p2 + p3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

# Update the print statement too
if settling_time is None:
    print(f"Done! Wrote data to {csv_filename}")
    print(f"Performance Metrics - Settling Time: Never settles, Overshoot: {overshoot:.2f}%")
else:
    print(f"Done! Wrote data to {csv_filename}")
    print(f"Performance Metrics - Settling Time: {settling_time:.2f}s, Overshoot: {overshoot:.2f}%")
print("Plot displayed successfully!")