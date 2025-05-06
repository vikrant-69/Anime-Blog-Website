import requests
import os
import shutil
import subprocess
import time
import sys
from pywinauto.application import Application
from pywinauto import timings

# --- Configuration ---
# !! IMPORTANT: MODIFY THESE VALUES !!
BINARY_URL = "YOUR_FIRMWARE_URL_HERE"  # <--- PASTE THE ACTUAL URL TO THE BINARY (.tar.md5 or similar)
TARGET_DIR = "C:\\Odin_Firmware"       # <--- SET YOUR DESIRED FOLDER FOR THE FINAL FIRMWARE FILE
ODIN_EXE_PATH = "C:\\path\\to\\Odin3 vX.XX.exe" # <--- SET THE FULL PATH TO YOUR Odin3.exe
ADB_EXE_PATH = "adb"                    # Set full path if adb is not in system PATH, e.g., "C:\\platform-tools\\adb.exe"

# pywinauto timings (increase if automation seems too fast)
timings.Timings.window_find_timeout = 15 # seconds to wait for a window
timings.Timings.cpu_usage_timeout = 5    # seconds for cpu stability before action

# Odin specific control identifiers (These might change between Odin versions!)
# Use tools like Inspect.exe (Windows SDK) or pywinauto's print_control_identifiers()
# to find the correct identifiers for *your* Odin version.
ODIN_WINDOW_TITLE_REGEX = r"Odin3.*" # Regex to match the Odin window title
# Common buttons (examples - ADJUST BASED ON YOUR ODIN):
# Check buttons like BL, AP, CP, CSC might have different 'auto_id' or 'control_id'
AP_BUTTON_IDENTIFIER = {"title": "AP", "control_type": "Button"} # Common identifier
# START_BUTTON_IDENTIFIER = {"title": "Start", "control_type": "Button"} # Common identifier
START_BUTTON_IDENTIFIER = "Start" # Sometimes simple title works for buttons

# File selection dialog identifiers (usually standard Windows)
FILE_DIALOG_TITLE = "Open"
FILE_DIALOG_FILENAME_EDIT_IDENTIFIER = "Edit1" # Often "Edit1" or {"class_name": "Edit", "control_id": 1148}
FILE_DIALOG_OPEN_BUTTON_IDENTIFIER = {"title": "Open", "control_type": "Button", "control_id": 1} # Often control_id 1
# --- End Configuration ---

def download_file(url, download_folder):
    """Downloads a file from a URL to the specified folder."""
    try:
        print(f"Attempting to download from: {url}")
        # Ensure download folder exists
        os.makedirs(download_folder, exist_ok=True)

        # Get filename from URL or Content-Disposition header
        response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        filename = ""
        if "content-disposition" in response.headers:
            cd = response.headers['content-disposition']
            filename = cd.split('filename=')[-1].strip('"')
        if not filename:
            filename = url.split('/')[-1]
        if not filename: # Fallback if URL is weird
             filename = "downloaded_firmware"

        download_path = os.path.join(download_folder, filename)
        print(f"Downloading to: {download_path}")

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192 # 8KB
        wrote = 0
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(block_size):
                wrote += len(chunk)
                f.write(chunk)
                if total_size > 0:
                    done = int(50 * wrote / total_size)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {wrote / (1024 * 1024):.2f}/{total_size / (1024 * 1024):.2f} MB")
                    sys.stdout.flush()
        print("\nDownload complete.")
        return download_path
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading file: {e}")
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred during download: {e}")
        return None

def move_and_rename(source_path, target_dir):
    """Moves the file, removes .md5 extension if present, and returns the new path."""
    try:
        if not os.path.exists(source_path):
            print(f"Error: Source file not found at {source_path}")
            return None

        filename = os.path.basename(source_path)
        base_name, ext = os.path.splitext(filename)

        # Remove .md5 if it's the final extension
        if ext.lower() == '.md5':
            new_filename = base_name
        else:
            new_filename = filename # Keep original name if no .md5

        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, new_filename)

        print(f"Moving '{source_path}' to '{target_path}'")
        shutil.move(source_path, target_path)
        print("Move and rename successful.")
        return target_path
    except Exception as e:
        print(f"Error moving/renaming file: {e}")
        return None

def run_command(command_parts):
    """Runs a command line command and prints output."""
    try:
        print(f"Running command: {' '.join(command_parts)}")
        result = subprocess.run(command_parts, capture_output=True, text=True, check=False, shell=False)
        print("STDOUT:")
        print(result.stdout if result.stdout else "<No stdout>")
        print("STDERR:")
        print(result.stderr if result.stderr else "<No stderr>")
        if result.returncode != 0:
            print(f"Warning: Command exited with non-zero status: {result.returncode}")
        return result.returncode == 0
    except FileNotFoundError:
        print(f"Error: Command not found (is '{command_parts[0]}' in your PATH or correctly specified?)")
        return False
    except Exception as e:
        print(f"Error running command {' '.join(command_parts)}: {e}")
        return False

def automate_odin(odin_path, firmware_path):
    """Starts Odin, loads firmware, and clicks Start using pywinauto."""
    try:
        print("Starting Odin...")
        # Use start() instead of connect() to launch the application
        app = Application(backend="uia").start(odin_path) # uia backend is generally preferred
        time.sleep(5) # Give Odin some time to load

        # Connect to the Odin window
        print("Connecting to Odin window...")
        odin_window = app.window(title_re=ODIN_WINDOW_TITLE_REGEX)
        odin_window.wait('visible', timeout=timings.Timings.window_find_timeout) # Wait until window is ready
        print("Odin window found.")
        # odin_window.print_control_identifiers() # Uncomment this to DEBUG identifiers

        # --- Interact with Odin ---
        # 1. Click the AP (or relevant) button
        print(f"Clicking AP button (using identifier: {AP_BUTTON_IDENTIFIER})...")
        ap_button = odin_window.child_window(**AP_BUTTON_IDENTIFIER)
        ap_button.wait('enabled', timeout=10) # Wait for button to be clickable
        ap_button.click_input() # Use click_input() for more human-like interaction
        print("AP button clicked.")
        time.sleep(1) # Short pause

        # 2. Handle the File Open Dialog
        print("Handling file open dialog...")
        file_dialog = app.window(title=FILE_DIALOG_TITLE)
        file_dialog.wait('visible', timeout=timings.Timings.window_find_timeout)
        print("File dialog found.")

        # Set the filename
        print(f"Setting filename: {firmware_path}")
        # file_dialog.print_control_identifiers() # Uncomment this to DEBUG identifiers
        filename_edit = file_dialog.child_window(best_match=FILE_DIALOG_FILENAME_EDIT_IDENTIFIER)
        filename_edit.wait('enabled', timeout=10)
        filename_edit.set_edit_text(firmware_path) # Use set_edit_text for reliability
        time.sleep(0.5)

        # Click the 'Open' button
        print("Clicking 'Open' button in dialog...")
        open_button = file_dialog.child_window(**FILE_DIALOG_OPEN_BUTTON_IDENTIFIER)
        open_button.wait('enabled', timeout=10)
        open_button.click_input()
        print("File selected.")
        time.sleep(2) # Give Odin time to load/verify the file

        # 3. Click the Start button in Odin
        print("Waiting for Odin to be ready for Start...")
        # Add checks here if needed (e.g., wait for "Added!!" message in Odin's log box)
        # This requires identifying the log box control.
        time.sleep(5) # Simple wait for file processing

        print(f"Clicking Start button (using identifier: {START_BUTTON_IDENTIFIER})...")
        start_button = odin_window.child_window(best_match=START_BUTTON_IDENTIFIER) # Use best_match for flexibility
        start_button.wait('enabled', timeout=30) # Wait longer, Odin might be busy
        start_button.click_input()
        print("Start button clicked. Flashing process initiated.")
        print("!!! MONITOR THE ODIN WINDOW FOR PROGRESS AND SUCCESS/FAILURE !!!")
        print("Automation will wait for a fixed time, but manual monitoring is essential.")

        # --- Waiting for Completion (Difficult Part) ---
        # Reliable waiting is hard. Odin doesn't offer a simple completion signal.
        # Options:
        # 1. Fixed Wait: Easiest, but unreliable.
        # 2. Monitor Log Box: Look for "PASS!" or "FAIL!" text. Requires identifying the log control.
        # 3. Monitor Window Title/Elements: Check if the "PASS" indicator appears.
        # We'll use a long fixed wait and recommend manual monitoring.
        wait_time_seconds = 600 # 10 minutes - ADJUST AS NEEDED
        print(f"Waiting for {wait_time_seconds} seconds for flashing to potentially complete...")
        time.sleep(wait_time_seconds)
        print("Wait time finished. Assuming flashing is done (check Odin!).")

        # Optional: Try to close Odin (can fail if flashing is stuck)
        try:
             print("Attempting to close Odin...")
             odin_window.close()
             time.sleep(2)
        except Exception as close_err:
             print(f"Could not automatically close Odin (may still be running): {close_err}")

        return True

    except timings.TimeoutError:
        print("Error: Timed out waiting for a window or control in Odin.")
        print("Possible causes: Incorrect identifiers, Odin version mismatch, Odin not responding.")
        # odin_window.print_control_identifiers() # Helps debug
        return False
    except Exception as e:
        print(f"An error occurred during Odin automation: {e}")
        # Try printing identifiers if a window object exists
        try:
            if 'odin_window' in locals() and odin_window.exists():
               print("Attempting to print Odin control identifiers for debugging:")
               odin_window.print_control_identifiers()
        except Exception as pe:
            print(f"Could not print control identifiers: {pe}")
        return False

# --- Main Script Execution ---
if __name__ == "__main__":
    print("--- Android Flashing Automation Script ---")

    # Validate Configuration
    if "YOUR_FIRMWARE_URL_HERE" in BINARY_URL:
        print("Error: Please update the BINARY_URL variable in the script.")
        sys.exit(1)
    if not os.path.exists(ODIN_EXE_PATH):
        print(f"Error: Odin executable not found at '{ODIN_EXE_PATH}'. Please update the path.")
        sys.exit(1)
    if not os.path.exists(TARGET_DIR):
         print(f"Warning: Target directory '{TARGET_DIR}' does not exist. It will be created.")


    # Step 1: Download the binary
    print("\n[Step 1/7] Downloading Firmware...")
    user_download_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    downloaded_path = download_file(BINARY_URL, user_download_folder)
    if not downloaded_path:
        print("Exiting due to download failure.")
        sys.exit(1)

    # Step 2 & 3: Move and Rename
    print("\n[Step 2 & 3/7] Moving and Renaming Firmware...")
    final_firmware_path = move_and_rename(downloaded_path, TARGET_DIR)
    if not final_firmware_path:
        print("Exiting due to file move/rename failure.")
        sys.exit(1)

    # Step 4: Reboot device to Download mode
    print("\n[Step 4/7] Rebooting device to Download Mode...")
    print("Ensure your device is connected via USB and USB Debugging is enabled.")
    input("Press Enter when ready to reboot device...") # Pause for user confirmation
    if not run_command([ADB_EXE_PATH, "reboot", "download"]):
         print("Warning: Failed to execute adb reboot download. Ensure device is connected and adb works.")
         print("Please manually put the device into Download Mode.")
         input("Press Enter once the device is in Download Mode...")
    else:
         print("Command sent. Waiting for device to enter Download mode (approx 15-30 seconds)...")
         time.sleep(25)

    # Step 5 & 6: Open Odin and Automate Flashing
    print("\n[Step 5 & 6/7] Starting Odin and Flashing Process...")
    if not automate_odin(ODIN_EXE_PATH, final_firmware_path):
        print("Error during Odin automation. Please check the Odin window and flash manually if needed.")
        print("Exiting script.")
        sys.exit(1)
    else:
        print("Odin automation part finished (or timed out).")

    # Step 7: Check for device after flashing
    print("\n[Step 7/7] Checking for device post-flashing...")
    print("Waiting for device to potentially reboot (approx 60 seconds)...")
    time.sleep(60)
    print("Running 'adb devices'...")
    run_command([ADB_EXE_PATH, "devices"])

    print("\n--- Automation Script Finished ---")
    print("Please verify the flashing result on your device and in the Odin log.")
    
