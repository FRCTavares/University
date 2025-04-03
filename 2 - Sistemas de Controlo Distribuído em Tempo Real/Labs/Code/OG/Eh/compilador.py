"""
File: compilador.py
Description: This script compiles and uploads the OG.ino sketch to multiple Raspberry Pi Pico 
devices on a Windows system. It supports deploying a distributed lighting control network by
assigning unique node IDs to each Pico device.

Usage: 
  python compilador.py
  python compilador.py --assign-ids
"""

import os
import subprocess
import glob
import shutil
import argparse
import serial.tools.list_ports
import time

# Project settings
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INO_FILE = os.path.join(PROJECT_DIR, "OG.ino")
BUILD_DIR = os.path.join(PROJECT_DIR, "build")

def compile_program(ino_file):
    """Compile the program using Arduino CLI"""
    print(f"Compiling {os.path.basename(ino_file)}...")
    
    # Create build directory if it doesn't exist
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)
    
    # Compile command for Arduino CLI
    compile_cmd = [
        "arduino-cli",
        "compile",
        "--fqbn",
        "rp2040:rp2040:rpipico",
        ino_file,
        "--output-dir",
        BUILD_DIR,
        "--verbose"
    ]
    
    try:
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print("Compilation successful!")
        return os.path.join(BUILD_DIR, "OG.ino.uf2")
    except subprocess.CalledProcessError as e:
        print("Compilation failed!")
        print(f"Error: {e.stderr}")
        return None

def find_pico_devices():
    """Find all connected Raspberry Pi Pico devices on Windows"""
    print("Scanning for connected Pico devices...")
    
    # List all COM ports
    ports = list(serial.tools.list_ports.comports())
    
    # Filter for Raspberry Pi Pico devices - more lenient detection
    pico_ports = []
    for port in ports:
        # Check if it's one of our known Pico COM ports
        if port.device in ['COM3', 'COM4', 'COM6'] or \
           "USB Serial Device" in port.description or \
           "RP2040" in port.description or \
           "USB Serial" in port.description:
            pico_ports.append(port)
    
    if not pico_ports:
        print("No Raspberry Pi Pico devices found. Please connect at least one Pico.")
        return []
    
    print(f"Found {len(pico_ports)} Pico devices:")
    for i, port in enumerate(pico_ports):
        print(f"  {i+1}. {port.device} - {port.description}")
    
    return pico_ports

def upload_to_pico(port, uf2_file, node_id=None):
    """Upload the compiled program to a specific Pico device"""
    print(f"\nUploading to {port.device} ({port.description})...")
    
    if node_id is not None:
        print(f"  Assigning Node ID: {node_id}")
        
        # Create a temporary copy of the INO file
        temp_ino_file = os.path.join(BUILD_DIR, f"OG_node{node_id}.ino")
        
        # Read the original file
        with open(INO_FILE, 'r') as f:
            code = f.read()
        
        # Replace or add node ID definition
        if "deviceConfig.nodeId =" in code:
            # Replace existing node ID assignment
            code_lines = code.split('\n')
            for i, line in enumerate(code_lines):
                if "deviceConfig.nodeId =" in line:
                    code_lines[i] = f"  deviceConfig.nodeId = {node_id};  // Auto-assigned by compilador.py"
                    break
            code = '\n'.join(code_lines)
        else:
            # If no assignment found, look for setup() function and add it there
            setup_pos = code.find("void setup()")
            if setup_pos != -1:
                # Find the opening brace of setup()
                brace_pos = code.find("{", setup_pos)
                if brace_pos != -1:
                    # Insert after the opening brace
                    code = code[:brace_pos+1] + f"\n  deviceConfig.nodeId = {node_id};  // Auto-assigned by compilador.py\n" + code[brace_pos+1:]
        
        # Write to temporary file
        with open(temp_ino_file, 'w') as f:
            f.write(code)
        
        # Compile the modified file
        temp_uf2 = compile_program(temp_ino_file)
        if not temp_uf2:
            return False
        
        # Use the temp file's path for upload
        uf2_file = temp_uf2
    
    # For actual upload, use the Arduino CLI
    upload_cmd = [
        "arduino-cli",
        "upload",
        "-p",
        port.device,
        "--fqbn",
        "rp2040:rp2040:rpipico",
        "--input-file",
        uf2_file
    ]
    
    try:
        result = subprocess.run(upload_cmd, check=True, capture_output=True, text=True)
        print(f"  Upload to {port.device} successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Upload to {port.device} failed!")
        print(f"  Error: {e.stderr}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compile and upload to multiple Pico devices")
    parser.add_argument("--assign-ids", action="store_true", help="Assign unique node IDs to each Pico")
    args = parser.parse_args()
    
    # Step 1: Compile the program
    uf2_file = compile_program(INO_FILE)
    if not uf2_file:
        return
    
    # Step 2: Find connected Pico devices
    pico_devices = find_pico_devices()
    if not pico_devices:
        return
    
    # Step 3: Upload to each device
    successful_uploads = 0
    # Use your specific node IDs instead of sequential numbers
    node_ids = [33, 40, 52]
    
    if len(pico_devices) > len(node_ids):
        print(f"Warning: Found {len(pico_devices)} devices but only have {len(node_ids)} node IDs.")
    
    for i, device in enumerate(pico_devices):
        # Only assign IDs up to the length of our node_ids list
        node_id = node_ids[i] if args.assign_ids and i < len(node_ids) else None
        if upload_to_pico(device, uf2_file, node_id):
            successful_uploads += 1
    
    print(f"\nUploaded to {successful_uploads} out of {len(pico_devices)} devices")
    
    # Step 4: Clean up
    print("\nCleaning up build files...")
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    
    print("\nDone! Your distributed network is ready for testing.")

if __name__ == "__main__":
    main()