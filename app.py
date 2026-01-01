
import cv2
import numpy as np
import threading
import time
from flask import Flask, Response
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)

# --- CONFIGURATION ---
HEF_PATH = "yolov8n_talos.hef"
CONF_THRESHOLD = 0.3
IO_QUEUE_SIZE = 1
CLASSES = ["Person", "Car", "Bicycle", "Motorcycle", "Heavy Vehicle"]

# --- GLOBAL SHARED STATE ---
# We use a global variable to pass the image from the AI thread to the Flask thread.
latest_frame = None

# A 'Lock' is a safety mechanism. It ensures that the AI thread doesn't write 
# to 'latest_frame' at the exact same moment the Flask thread tries to read it.
# This prevents corrupted images or crashes ("Race Conditions").
lock = threading.Lock()

app = Flask(__name__)

# --- 1. UTILITIES ---
# (Same logic as the synchronous version, just condensed for brevity)
def preprocess_image(image, target_size=640):
    """Resize with letterboxing (adding gray padding)."""
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (nw, nh))
    
    # Fill canvas with 114 (gray)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_w = (target_size - nw) // 2
    pad_h = (target_size - nh) // 2
    canvas[pad_h:pad_h+nh, pad_w:pad_w+nw, :] = resized_image
    
    return canvas, scale, (pad_w, pad_h)

def post_process_and_draw(raw_results, scale, pads, orig_dims, frame):
    pad_w, pad_h = pads
    
    # 1. Get the raw output batch
    # The logs showed this is a list of length 1: [[Class0, Class1, Class2...]]
    batch = list(raw_results.values())[0]

    # --- THE FIX: UNWRAP THE BATCH LIST ---
    # We want the inner list of 5 items, not the outer list of 1 item.
    if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], list):
        detections_by_class = batch[0]
    else:
        detections_by_class = batch
    # ---------------------------------------

    boxes_list = []
    scores_list = []
    class_ids_list = []

    # 2. Iterate through each class (Now correctly loops 5 times)
    for class_id, detections in enumerate(detections_by_class):
        
        # Safety: If detections is None or empty
        if detections is None:
            continue
            
        # Safety: Force to numpy array if it's a list
        if isinstance(detections, list):
            detections = np.array(detections)
            
        # Safety: Check size
        if detections.size == 0:
            continue

        # Handle "Flat" Single Detections (1D -> 2D)
        if detections.ndim == 1:
            detections = np.expand_dims(detections, axis=0)
            
        for det in detections:
            # Ensure row has 5 elements (x, y, w, h, score)
            if det.size < 5:
                continue

            score = det[4]

            # OPTIMIZATION: Filter by confidence immediately
            if score < 0.25: # Use your CONF_THRESHOLD here
                continue

            # 3. Denormalize coordinates (Fix for "Messed Up" Boxes)
            # Hailo models often output [ymin, xmin, ymax, xmax] instead of [x, y, w, h]
            # We explicitly unpack the 4 numbers to check
            
            box_raw = det[:4]
            # Assuming format is [ymin, xmin, ymax, xmax] - Standard for Hailo
            ymin, xmin, ymax, xmax = box_raw

            # Convert to [x, y, w, h] (Center format) for the rest of your code
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + w / 2
            y = ymin + h / 2
            
            # Now scale to 640
            box = np.array([x, y, w, h]) * 640
            
            boxes_list.append(box)
            scores_list.append(score)
            class_ids_list.append(class_id)
    
    if not boxes_list:
        return frame

    # Convert to numpy for vector math
    boxes_xywh = np.array(boxes_list)
    scores = np.array(scores_list)
    class_ids = np.array(class_ids_list)

    # 4. Convert coordinates back to original image space
    x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    
    x1 = (x - w/2 - pad_w) / scale
    y1 = (y - h/2 - pad_h) / scale
    x2 = (x + w/2 - pad_w) / scale
    y2 = (y + h/2 - pad_h) / scale

    # 5. Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(
        bboxes=[[int(xx), int(yy), int(ww), int(hh)] for xx, yy, ww, hh in zip(x1, y1, x2-x1, y2-y1)],
        scores=scores.tolist(),
        score_threshold=0.25, # Use your CONF_THRESHOLD
        nms_threshold=0.45
    )

    # 6. Draw boxes
    for i in indices:
        idx = i if isinstance(i, (int, np.integer)) else i[0]
        
        bx1, by1 = int(x1[idx]), int(y1[idx])
        bx2, by2 = int(x2[idx]), int(y2[idx])
        cls_id = int(class_ids[idx])
        
        # Check bounds to prevent crashing on lookup
        if 0 <= cls_id < len(CLASSES):
             label_text = CLASSES[cls_id]
        else:
             label_text = f"Class {cls_id}"

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label_text} {scores[idx]:.2f}", (bx1, by1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# --- 2. THE AI WORKER THREAD ---
# This function runs in the background, independently of the web server.
# Its ONLY job is to update 'latest_frame' as fast as possible.
def ai_worker():
    global latest_frame
    print("AI Thread: Initializing Hailo...")
    
    # Init Hailo
    try:
        target = VDevice()
        hef = HEF(HEF_PATH)
        configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        
        input_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        input_info = hef.get_input_vstream_infos()[0]
    except Exception as e:
        print(f"AI Thread: Failed to init Hailo. Error: {e}")
        return

    # FIXED PIPELINE:
    # 1. Ask for FULL resolution first (1456x1088 for IMX296) to get the wide angle.
    # 2. Then use 'videoscale' to shrink it down to 640x480 for the AI.
    gst_pipeline = (
        "libcamerasrc ! "
        "video/x-raw, format=NV12, width=1456, height=1088 ! "  # <--- FORCE FULL SENSOR
        "videoscale ! "
        "video/x-raw, width=640, height=480 ! "                # <--- NOW RESIZE
        "videoflip method=rotate-180 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("AI Thread: CRITICAL ERROR - Camera failed to open.")
        return

    print("AI Thread: Starting Loop...")
    
    try:
        # --- NEW: EXPLICITLY ACTIVATE THE NETWORK GROUP FIRST ---
        with network_group.activate(): 
            # NOW open the streams inside the active group
            with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("AI Thread: Camera read failed.")
                        time.sleep(1)
                        continue

                    processed, scale, pads = preprocess_image(frame, target_size=640)
                    
                    input_data = {input_info.name: np.expand_dims(processed, axis=0)}
                    
                    # This line should now succeed
                    raw_results = infer_pipeline.infer(input_data)
                    
                    final_frame = post_process_and_draw(raw_results, scale, pads, frame.shape[:2], frame)
                    
                    with lock:
                        latest_frame = final_frame.copy()
                        
    except Exception as e:
        print(f"\nCRITICAL HAILO ERROR: {e}")
        # If this persists, the HEF might be compiled for the wrong chip (Hailo-8 vs Hailo-8L)

# --- 3. FLASK SERVER ---
@app.route('/')
def index():
    # Simple HTML page to host the video stream
    return '<html><body><h1>TALOS View</h1><img src="/video_feed" width="100%"></body></html>'

@app.route('/video_feed')
def video_feed():
    # This function handles the *viewing* of the data.
    # It runs in its own thread managed by Flask.
    def generate():
        while True:
            # Check for new frame
            with lock:
                if latest_frame is None:
                    continue # Wait until AI thread produces the first frame
                
                # Make a local reference so we can release the lock fast
                frame_to_encode = latest_frame
                
            # Encode to JPEG
            # We do this OUTSIDE the lock to keep the lock held for as short as possible
            ret, buffer = cv2.imencode('.jpg', frame_to_encode)
            
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Sleep slightly to control the web stream frame rate (approx 30 FPS).
            # This saves CPU; otherwise, this loop would spin as fast as possible,
            # encoding the same frame over and over.
            time.sleep(0.03) 

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # 1. Start the AI Worker Thread
    # daemon=True means this thread will automatically die when the main program exits.
    t = threading.Thread(target=ai_worker, daemon=True)
    t.start()
    
    # 2. Start the Web Server
    # threaded=True allows Flask to handle multiple browser tabs if needed
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
