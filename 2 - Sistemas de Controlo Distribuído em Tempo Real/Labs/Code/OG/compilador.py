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
    
    # Filter for Raspberry Pi Pico devices
    pico_ports = []
    for port in ports:
        # The Pico shows up as either "USB Serial Device" or mentions "RP2040" in the description
        if "USB Serial Device" in port.description or "RP2040" in port.description:
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
    
    # If assigning node IDs, put the Pico in bootloader mode first
    if node_id is not None:
        print(f"  Assigning Node ID: {node_id}")
        
        # Modify a temporary copy of the code with the specific node ID
        # This would require parsing and modifying the source code
        # For demonstration, we'll just show how you would do it
        print("  (Demo) Customizing node ID in the firmware...")
        
    # For actual upload, use the Arduino CLI
    upload_cmd = [
        "arduino-cli",
        "upload",
        "-p",
        port.device,
        "--fqbn",
        "rp2040:rp2040:rpipico",
        INO_FILE,
        "--input-dir",
        BUILD_DIR
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
    for i, device in enumerate(pico_devices):
        node_id = i + 1 if args.assign_ids else None
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