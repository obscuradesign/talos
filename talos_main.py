from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# --- 1. LOAD THE BRAIN ---
# This loads the standard YOLO model. 
# If you have a custom trained model, change 'yolov8n.pt' to your filename.
print("Loading AI Model...")
model = YOLO("yolov8n_ncnn_model")
print("Model Loaded!")

# --- 2. SETUP THE CAMERA (The Working GStreamer Pipeline) ---
gst_pipeline = (
    "libcamerasrc ! "
    "video/x-raw,format=YUY2,width=1456,height=1088 ! "
    "videoconvert ! "
    "appsink drop=1"
)

camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

def generate_frames():
    if not camera.isOpened():
        print("CRITICAL ERROR: Camera is NOT open!")
        return

    print("Starting video loop...") # Debug print

    while True:
        # print("Reading camera...")  # Commented out to avoid spam, uncomment if needed
        success, frame = camera.read()
        if not success:
            print("ERROR: Failed to read frame.")
            break
        
        # Resize
        frame_resized = cv2.resize(frame, (640, 480))
        
        # --- THE SUSPECT: AI INFERENCE ---
        print("Running AI...", end="", flush=True) 
        results = model(frame_resized, task='detect') # Added task='detect' to fix warning
        print("Done!")
        
        # Draw boxes
        annotated_frame = results[0].plot()
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 0.0.0.0 listens on the TALOS_LINK hotspot
    app.run(host='0.0.0.0', port=5000)
