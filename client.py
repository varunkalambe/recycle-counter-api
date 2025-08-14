import cv2
import requests
import time
import os

# --- Configuration ---
# Replace with the IP address or URL of the machine running your FastAPI server
API_URL = "http://<YOUR_SERVER_IP_ADDRESS>:8000/process-frame/" 
API_KEY = "YOUR_SUPER_SECRET_API_KEY"  # Must match the key in main.py
FRAME_INTERVAL = 2  # Time in seconds between sending frames

# --- Main Program ---
headers = {
    "X-API-KEY": API_KEY
}

# Initialize the camera
cap = cv2.VideoCapture(0) # 0 is the default camera

if not cap.isOpened():
    print("🔥 Error: Could not open camera.")
    exit()

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
            response = requests.post(API_URL, headers=headers, files=files, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
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