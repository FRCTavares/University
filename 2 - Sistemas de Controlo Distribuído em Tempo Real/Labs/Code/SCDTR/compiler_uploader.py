"""
File: compiler_uploader.py
Description: This script compiles and uploads an Arduino program to Raspberry Pi Pico microcontroller boards
using the Arduino CLI on Windows. The script targets specific COM ports (COM3, COM4, COM6) for uploading.

Date: April 9, 2025
"""

import os
import subprocess
import shutil
import time
import glob
import sys

# Set the path to the .ino file
ino_file = "SCDTR.ino"  # Updated to use SCDTR.ino in the current directory

def compile_program(ino_file):
    build_dir = "build"

    # Create the build directory if it does not exist
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)

    print(f"Compiling {ino_file}...")
    # Compile the program using the Arduino CLI
    compile_cmd = [
        "arduino-cli",
        "compile",
        "--fqbn",
        "rp2040:rp2040:rpipico",
        ino_file,
        "--output-dir",
        build_dir,
    ]
    subprocess.run(compile_cmd, check=True)

    # Return the path to the compiled program
    return f"{build_dir}/SCDTR.ino.uf2"

def get_pico_devices():
    # Define the specific COM ports for the three Picos
    return ["COM3", "COM4", "COM6"]

def wait_for_bootloader_drive():
    """Wait for a Pico bootloader drive to appear"""
    print("Waiting for bootloader drive...")
    
    # Check for RPI-RP2 drive every second for up to 30 seconds
    for _ in range(30):
        # Look for drives with RPI-RP2 name
        drives = [d for d in glob.glob("?:/") if os.path.exists(f"{d}INFO_UF2.TXT")]
        if drives:
            print(f"Found bootloader drive at {drives[0]}")
            return drives[0]
        time.sleep(1)
    
    return None

def upload_to_pico_direct(uf2_path):
    """Upload directly by copying to the bootloader drive"""
    bootloader_drive = wait_for_bootloader_drive()
    
    if not bootloader_drive:
        print("No bootloader drive found. Please manually put the Pico in bootloader mode.")
        return False
    
    try:
        # Copy the UF2 file to the Pico drive
        shutil.copy(uf2_path, bootloader_drive)
        print(f"UF2 file copied to {bootloader_drive}")
        time.sleep(2)  # Wait for reset
        return True
    except Exception as e:
        print(f"Error copying UF2 file: {e}")
        return False

def upload_to_pico(device, ino_file, compiled_uf2):
    print(f"Preparing to upload to {device}...")
    
    # First attempt using Arduino CLI
    try:
        print(f"Attempting standard upload to {device}...")
        upload_cmd = [
            "arduino-cli",
            "upload",
            "-p",
            device,
            "--fqbn",
            "rp2040:rp2040:rpipico",
            ino_file,
        ]
        subprocess.run(upload_cmd, check=True, timeout=10)
        print(f"Upload to {device} successful!")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print(f"Standard upload failed for {device}.")
    

    
    input("Press Enter when the Pico is in bootloader mode...")
    
    # Try direct upload by copying to the drive
    if upload_to_pico_direct(compiled_uf2):
        print(f"Upload to Pico (previously on {device}) successful!")
        return True
    else:
        print(f"Failed to upload to device {device}.")
        return False

def main():
    print("Starting the upload process for SCDTR.ino to three Pico boards...")

    # Compile the program
    compiled_program = compile_program(ino_file)
    
    # Get the target COM ports
    pico_devices = get_pico_devices()

    # Print a message indicating which devices the program will be uploaded to
    print(f"Targeting the following devices:")
    for device in pico_devices:
        print(f"  * {device}")

    # Upload the program to each Pico device
    success_count = 0
    for device in pico_devices:
        if upload_to_pico(device, ino_file, compiled_program):
            success_count += 1
        # Add a short pause between uploads
        time.sleep(1)

    print(f"Upload complete: {success_count} of {len(pico_devices)} devices programmed successfully.")

    # Clean up - remove the build directory after uploading
    if os.path.exists("build"):
        shutil.rmtree("build")
        print("Build directory cleaned up.")

if __name__ == "__main__":
    main()