import cv2
import requests
import time
import os
import io

# --- Configuration ---
# You must update this with your deployed server's public URL
API_ENDPOINT = "https://recycle-counter-api-1.onrender.com/process-frame/"
# This key must match the API_KEY environment variable on your server
API_KEY = "rPi-b0tt1e-sCAn-9zX7-qW3e"
FRAME_INTERVAL = 2

# --- New Configuration for FPS ---
DESIRED_FPS = 3  # Set your desired frames per second

# --- Main Program ---
headers = {
    "X-API-KEY": API_KEY  # This header name must match API_KEY_NAME in main.py
}

# Initialize the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("🔥 Error: Could not open camera.")
    exit()

# --- Set the desired FPS ---
cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

print("✅ Camera started. Will send a frame every {} seconds.".format(FRAME_INTERVAL))

try:
    while True:
        # 1. Capture a single frame
        ret, frame = cap.read()
        if not ret:
            print("🔥 Warning: Could not read frame from camera.")
            time.sleep(FRAME_INTERVAL)
            continue

        # 2. Encode the frame as a JPEG image in memory
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            print("🔥 Warning: Could not encode frame.")
            continue
        
        # Create a file-like object in memory
        io_buf = io.BytesIO(buffer)

        # 3. Prepare the file for the POST request
        files = {'file': ('frame.jpg', io_buf, 'image/jpeg')}

        # 4. Send the frame to the API
        try:
            response = requests.post(API_ENDPOINT, headers=headers, files=files, timeout=10)
            response.raise_for_status()
            
            print(f"✅ Frame sent successfully. Response: {response.json()}")

        except requests.exceptions.RequestException as e:
            print(f"🔥 Error sending frame: {e}")

        # 5. Wait for the next interval
        time.sleep(FRAME_INTERVAL)

except KeyboardInterrupt:
    print("\n🛑 Program stopped by user.")

finally:
    # Release the camera
    cap.release()
    print("✅ Camera released.")