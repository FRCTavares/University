import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# List of data files to process
data_files = ["greedy33.txt", "greedy52.txt", "greedy40.txt"]

# Function to parse data from a file
def parse_data_file(filename):
    data = {}
    
    # Check if file exists and is not empty
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print(f"Warning: File {filename} doesn't exist or is empty")
        return None
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines or comment lines
            if not line or line.startswith("//"):
                i += 1
                continue
            
            # If we encounter a data line with comma-separated values
            if "," in line:
                parts = line.split(',')
                key = parts[0]
                
                # Try to convert values to float if possible
                try:
                    values = [float(x) for x in parts[1:] if x.strip()]
                    data[key] = np.array(values)
                except ValueError:
                    # If conversion fails, store as strings
                    data[key] = np.array(parts[1:])
                    
                i += 1
            else:
                i += 1
    
    return data

# Process each data file
all_data = {}
for idx, filename in enumerate(data_files, 1):
    dataset = parse_data_file(filename)
    if dataset:
        # Use the filename without extension as the dataset name
        dataset_name = os.path.splitext(filename)[0]
        all_data[dataset_name] = dataset
        print(f"Successfully parsed {filename}")

# Define colors for different datasets
dataset_colors = {
    'greedy33': {'lux': '#2ca02c', 'setpoint': '#1f77b4', 'duty': '#ff7f0e', 'ext_lux': '#d62728'},
    'greedy52': {'lux': '#9467bd', 'setpoint': '#8c564b', 'duty': '#e377c2', 'ext_lux': '#7f7f7f'},
    'greedy40': {'lux': '#bcbd22', 'setpoint': '#17becf', 'duty': '#aec7e8', 'ext_lux': '#ffbb78'}
}

# Line styles for different datasets
dataset_styles = {'greedy33': '-', 'greedy52': '--', 'greedy40': '-.'}

# Process data for each dataset
for dataset_name, data in all_data.items():
    print(f"Processing {dataset_name} data...")
    
    # Map the data to variables (with fallbacks to empty arrays if not found)
    time_data = data.get('timestamp_ms', np.array([]))
    if len(time_data) == 0:
        print(f"Warning: No timestamp data found in {dataset_name}")
        continue
        
    measured_data = data.get('lux', np.array([]))
    duty_data = data.get('duty', np.array([]))
    setpoint_data = data.get('setpoint', np.array([]))
    flicker_data = data.get('flicker', np.array([]))
    energy_data = data.get('avg_energy', np.array([]))
    visibility_data = data.get('avg_vis_error', np.array([]))
    external_lux_data = data.get('ext_lux', np.array([]))
    avg_flicker_data = data.get('avg_flicker', np.array([]))

    # Calculate average metrics if needed
    non_zero_flicker = flicker_data[flicker_data > 0]
    avg_flicker = np.mean(non_zero_flicker) if len(non_zero_flicker) > 0 else 0
    
    # Store filtered data for this dataset
    if len(time_data) > 0:
        time_zero = time_data - time_data[0]
        # Filter data between 3-50 seconds (3000-50000 milliseconds)
        idx = (time_zero >= 3000) & (time_zero <= 50000)
        
        # Check if we have valid filtered data
        if np.any(idx):
            data['time_filtered'] = (time_zero[idx] / 1000) - 3  # Convert to seconds and re-zero to start at 0
            data['measured_filtered'] = measured_data[idx] if len(measured_data) > 0 else np.array([])
            data['duty_filtered'] = duty_data[idx] if len(duty_data) > 0 else np.array([])
            data['setpoint_filtered'] = setpoint_data[idx] if len(setpoint_data) > 0 else np.array([])
            data['visibility_filtered'] = visibility_data[idx] if len(visibility_data) > 0 else np.array([])
            data['energy_filtered'] = energy_data[idx] if len(energy_data) > 0 else np.array([])
            data['flicker_filtered'] = flicker_data[idx] if len(flicker_data) > 0 else np.array([])
            data['external_lux_filtered'] = external_lux_data[idx] if len(external_lux_data) > 0 else np.array([])
            data['avg_flicker'] = avg_flicker
        else:
            print(f"Warning: No data within the 3-50 second timeframe for {dataset_name}")

# 1. INDIVIDUAL DATASET PLOTS
for dataset_name, data in all_data.items():
    if 'time_filtered' not in data or len(data['time_filtered']) == 0:
        continue
        
    plt.figure(figsize=(15, 7), num=f"{dataset_name} - Control Parameters")
    ax1 = plt.gca()
    lines = []
    
    # Get dataset-specific colors
    dataset_color = dataset_colors.get(dataset_name, 
                                   {'lux': 'blue', 'setpoint': 'green', 'duty': 'red', 'ext_lux': 'purple'})
    
    # Plot setpoint and measured lux on primary y-axis
    p1 = ax1.plot(data['time_filtered'], data['setpoint_filtered'], 
                  color=dataset_color['setpoint'], linewidth=2, 
                  label='Reference Lux')
    
    p2 = ax1.plot(data['time_filtered'], data['measured_filtered'], 
                  color=dataset_color['lux'], linewidth=2, 
                  label='Measured Lux')
    
    lines.extend(p1 + p2)
    
    # Set axis labels and limits
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Illuminance [lx]')
    ax1.grid(True)
    ax1.set_ylim(0, 60)  # Adjust based on your data
    
    # Create secondary y-axis for duty cycle
    ax2 = ax1.twinx()
    
    # Plot duty cycle on secondary y-axis
    p3 = ax2.plot(data['time_filtered'], data['duty_filtered'], 
                  color=dataset_color['duty'], linewidth=2, linestyle='--',
                  label='Duty Cycle')
    
    lines.extend(p3)
    
    ax2.set_ylabel('Duty Cycle [ratio]')
    ax2.set_ylim(0, 1.1)
    
    # Create combined legend
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title(f"{dataset_name} - Reference, Measured Lux & Duty Cycle")
    plt.tight_layout()
    plt.show()

# 2. COMBINED PLOT - Lux and Duty Cycle for all datasets
plt.figure(figsize=(20, 8), num="Lux & DutyCycle - All Datasets")
ax1 = plt.gca()
lines = []

for dataset_name, data in all_data.items():
    if 'time_filtered' not in data or len(data['time_filtered']) == 0:
        continue
        
    dataset_color = dataset_colors.get(dataset_name, 
                                    {'lux': 'blue', 'setpoint': 'green', 'duty': 'red', 'ext_lux': 'purple'})
    line_style = dataset_styles.get(dataset_name, '-')
    
    # Plot setpoint and measured lux
    p1 = ax1.plot(data['time_filtered'], data['setpoint_filtered'], 
                  color=dataset_color['setpoint'], linestyle=line_style, linewidth=1.7, 
                  label=f'{dataset_name} - Reference Lux')
    
    p2 = ax1.plot(data['time_filtered'], data['measured_filtered'], 
                  color=dataset_color['lux'], linestyle=line_style, linewidth=1.7, 
                  label=f'{dataset_name} - Measured Lux')
    
    lines.extend(p1 + p2)
    
    # Plot external lux if available
    if 'external_lux_filtered' in data and len(data['external_lux_filtered']) > 0:
        p4 = ax1.plot(data['time_filtered'], data['external_lux_filtered'], 
                      color=dataset_color['ext_lux'], linestyle=line_style, linewidth=1.7, 
                      label=f'{dataset_name} - External Lux')
        lines.extend(p4)

# Create secondary y-axis for duty cycle
ax2 = ax1.twinx()

for dataset_name, data in all_data.items():
    if 'time_filtered' not in data or len(data['time_filtered']) == 0:
        continue
        
    dataset_color = dataset_colors.get(dataset_name, 
                                    {'lux': 'blue', 'setpoint': 'green', 'duty': 'red', 'ext_lux': 'purple'})
    line_style = dataset_styles.get(dataset_name, '-')
    
    # Plot duty cycle
    p3 = ax2.plot(data['time_filtered'], data['duty_filtered'], 
                  color=dataset_color['duty'], linestyle=line_style, linewidth=1.7, 
                  label=f'{dataset_name} - Duty Cycle')
    
    lines.extend(p3)

# Set axis labels and limits
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Illuminance [lx]')
ax1.grid(True)
ax1.set_ylim(0, 60)  # Adjust based on your data

ax2.set_ylabel('Duty Cycle [ratio]')
ax2.set_ylim(0, 1.1)  # Slightly increased to fully show the square wave

# Create combined legend
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')
plt.title("Reference, Measured & External Lux with LED Duty Cycles - All Datasets")
plt.tight_layout()
plt.show()

# 3. VISIBILITY ERROR AND ENERGY - Combined plot for all datasets
plt.figure(figsize=(20, 8), num="Visibility & Energy - All Datasets")
ax1 = plt.gca()
lines = []

for dataset_name, data in all_data.items():
    if 'time_filtered' not in data or len(data['time_filtered']) == 0 or 'visibility_filtered' not in data:
        continue
        
    dataset_color = dataset_colors.get(dataset_name, 
                                    {'lux': 'blue', 'setpoint': 'green', 'duty': 'red'})
    line_style = dataset_styles.get(dataset_name, '-')
    
    # Plot visibility error
    p1 = ax1.plot(data['time_filtered'], data['visibility_filtered'], 
                  color=dataset_color['lux'], linestyle=line_style, linewidth=1.7, 
                  label=f'{dataset_name} - Visibility Error')
    
    lines.extend(p1)

# Create secondary y-axis for energy
ax2 = ax1.twinx()

for dataset_name, data in all_data.items():
    if 'time_filtered' not in data or len(data['time_filtered']) == 0 or 'energy_filtered' not in data:
        continue
        
    dataset_color = dataset_colors.get(dataset_name, 
                                    {'lux': 'blue', 'setpoint': 'green', 'duty': 'red'})
    line_style = dataset_styles.get(dataset_name, '-')
    
    # Plot energy
    p2 = ax2.plot(data['time_filtered'], data['energy_filtered'], 
                  color=dataset_color['duty'], linestyle=line_style, linewidth=1.7, 
                  label=f'{dataset_name} - Energy')
    
    lines.extend(p2)

# Set axis labels
ax1.set_xlabel('Time (t) [s]')
ax1.set_ylabel('Avg Visibility Error (t) [lx]')  
ax1.grid(True)
ax2.set_ylabel('Avg Energy (t) [J]')

# Create combined legend
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')
plt.title("Average Visibility Error and Energy Consumption - All Datasets")
plt.tight_layout()
plt.show()

# 4. FLICKER - Combined plot for all datasets
plt.figure(figsize=(20, 8), num="Flicker - All Datasets")
ax = plt.gca()
lines = []

for dataset_name, data in all_data.items():
    if 'time_filtered' not in data or len(data['time_filtered']) == 0 or 'flicker_filtered' not in data:
        continue
        
    dataset_color = dataset_colors.get(dataset_name, 
                                    {'lux': 'blue', 'setpoint': 'green', 'duty': 'red'})
    line_style = dataset_styles.get(dataset_name, '-')
    
    # Plot flicker
    p = ax.plot(data['time_filtered'], data['flicker_filtered'], 
                color=dataset_color['setpoint'], linestyle=line_style, linewidth=1.7, 
                label=f'{dataset_name} - Flicker (Avg: {data.get("avg_flicker", 0):.6f})')
    
    lines.extend(p)
    
    # Plot average flicker line
    if "avg_flicker" in data:
        ax.axhline(y=data["avg_flicker"], color=p[0].get_color(), linestyle='--', alpha=0.5)

# Set axis labels
ax.set_xlabel('Time [s]')
ax.set_ylabel('Flicker (t) [s⁻¹]')
ax.grid(True)

# Create legend
ax.legend(loc='upper right')
plt.title("Flicker Over Time - All Datasets")
plt.tight_layout()
plt.show()