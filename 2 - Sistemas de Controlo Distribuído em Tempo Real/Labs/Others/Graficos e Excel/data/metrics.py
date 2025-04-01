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
    Searches 'whole_text' for a line starting with e.g. 'Time:' 
    and returns the list of floats after that colon.
    """
    pattern = rf"{label}:\s*(.*)"
    match = re.search(pattern, whole_text)
    if not match:
        return []
    floats_part = match.group(1)
    return [float(x.strip()) for x in floats_part.split(',') if x.strip()]

# 1) Read the entire text file
with open(txt_filename, "r") as f:
    content = f.read()

# 2) Parse each named array from the text
time_data          = np.array(parse_line("Time", content))
measured_data      = np.array(parse_line("MeasuredLux", content))
duty_data          = np.array(parse_line("DutyCycle", content))
setpoint_data      = np.array(parse_line("SetpointLux", content))
flicker_data       = np.array(parse_line("Flicker", content))
energy_data        = np.array(parse_line("Energy", content))
visibility_data    = np.array(parse_line("VisibilityError", content))
jitter_data        = np.array(parse_line("Jitter_us", content))
external_lux_data  = np.array(parse_line("ExternalLux", content))

# Calculate average metrics
avg_jitter = np.mean(jitter_data[jitter_data > 0]) / 1000000  # Convert to milliseconds
non_zero_flicker = flicker_data[flicker_data > 0]
avg_flicker = np.mean(non_zero_flicker) if len(non_zero_flicker) > 0 else 0

# After parsing data, create a new synthetic duty cycle function
def create_led_duty_cycle(time_data):
    """Creates a duty cycle for another LED with 20-second periods (ON/OFF)"""
    # Normalized time to identify 20-second periods
    period_time = (time_data - time_data[0]) / 1000  # Convert to seconds
    shift_amount = 1  # 2-second shift to the left
    shifted_time = period_time + shift_amount  # Add to shift left
    # Simple square wave with 10-second period (1 for first 5s, 0 for next 5s)
    led_duty = np.zeros_like(period_time)
    for i, t in enumerate(shifted_time):
        # If in the first half of the period (0-5s), set to 1
        if (t % 10) < 5:
            led_duty[i] = 1
        else:
            led_duty[i] = 0
    return led_duty

# Create the alternative duty cycle
#led_duty_data = create_led_duty_cycle(time_data)

# 4) Prepare data for plotting (filter out first 15s to 45s)
time_zero = time_data - time_data[0]
# Filter data between 15-45 seconds (15000-45000 milliseconds)
idx = (time_zero >= 3000) & (time_zero <= 50000)
time_filtered        = (time_zero[idx] / 1000) - 3  # Re-zero to start at 0
measured_filtered    = measured_data[idx]
duty_filtered        = duty_data[idx]
setpoint_filtered    = setpoint_data[idx]
visibility_filtered  = visibility_data[idx]
energy_filtered      = energy_data[idx]
flicker_filtered     = flicker_data[idx]
jitter_filtered      = jitter_data[idx]/1000000  
#external_lux_filtered = external_lux_data[idx]
#led_duty_filtered = led_duty_data[idx]

# Plot 1: Lux (measured & setpoint) and Duty Cycle
plt.figure(figsize=(20, 6), num="Lux & DutyCycle")
ax1 = plt.gca()
p1 = ax1.plot(time_filtered, setpoint_filtered, '#1f77b4', linewidth=1.7, label='Reference Lux')  # Deep blue
p2 = ax1.plot(time_filtered, measured_filtered, '#2ca02c', linewidth=1.7, label='Measured Lux')    # Green
#p4 = ax1.plot(time_filtered, external_lux_filtered, '#d62728', linewidth=1.7, label='External Lux') # Red
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Illuminance [lx]')
ax1.grid(True)
ax1.set_ylim(0, 60)  # Adjusted based on your data

ax2 = ax1.twinx()
p3 = ax2.plot(time_filtered, duty_filtered, '#ff7f0e', linewidth=1.7, label='LED 1 Duty Cycle')        # Orange
#p5 = ax2.plot(time_filtered, led_duty_filtered, '#9467bd', linewidth=1.7, label='LED 2 Duty Cycle')    # Purple
ax2.set_ylabel('Duty Cycle [ratio]')
ax2.set_ylim(0, 1.1)  # Slightly increased to fully show the square wave

lines = p1 + p2 + p3 #+ p3 #+ p5
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')
plt.title("Reference, Measured & External Lux with LED Duty Cycles")
plt.tight_layout()
plt.show()

'''
# Plot 2: Visibility Error and Energy
plt.figure(figsize=(20, 6), num="Visibility & Energy")
ax1 = plt.gca()
p1 = ax1.plot(time_filtered, visibility_filtered, 'm-', linewidth=1.7, label='Cumulative Visibility Error')
ax1.set_xlabel('Time (t) [s]')
ax1.set_ylabel('Σ Visibility Error (t) [lx·s]')  # Added sigma notation
ax1.grid(True)
ax2 = ax1.twinx()
p2 = ax2.plot(time_filtered, energy_filtered, 'c-', linewidth=1.7, label='Cumulative Energy')
ax2.set_ylabel('Σ Energy (t) [J]')  # Added sigma notation
lines = p1 + p2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')
plt.title("Cumulative Visibility Error and Energy Consumption")
plt.tight_layout()
plt.show()

# Plot 3: Flicker
plt.figure(figsize=(10, 6), num="Flicker")
plt.plot(time_filtered, flicker_filtered, 'k-', linewidth=1.7, label='Flicker')
plt.axhline(y=avg_flicker, color='r', linestyle='--', label=f'Average: {avg_flicker:.6f}')
plt.xlabel('Time[s]')
plt.ylabel('Flicker (t) [s⁻¹]')
plt.grid(True)
plt.title(f"Flicker Over Time (Avg: {avg_flicker:.6f})")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 4: Jitter
plt.figure(figsize=(10, 6), num="Jitter")
plt.plot(time_filtered, jitter_filtered, 'b-', linewidth=1.7, label='Jitter')
plt.axhline(y=avg_jitter, color='r', linestyle='--', label=f'Average: {avg_jitter:.2f} ms')
plt.xlabel('Time[s]')
plt.ylabel('Jitter (t) [ms]')
plt.grid(True)
plt.title(f"System Jitter (Avg: {avg_jitter:.2f} ms)")
plt.legend()
plt.tight_layout()
plt.show()
'''