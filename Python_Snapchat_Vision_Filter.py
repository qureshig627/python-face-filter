import cv2
import numpy as np
import dlib
import tkinter as tk
from tkinter import filedialog
import time
import math
from collections import deque

def upload_file(title, filetypes):
    """Opens a file dialog to select a file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path

class Stabilizer:
    """
    A simple class to smooth out values (like position or angle) over time.
    It uses a moving average of the last 'n' measurements.
    """
    def __init__(self, size=5):
        self.buffer = deque(maxlen=size)

    def update(self, measurement):
        """Adds a new measurement and returns the smoothed value."""
        self.buffer.append(measurement)
        # Calculate the average of the buffer
        smoothed_value = np.mean(self.buffer, axis=0)
        return smoothed_value

def rotate_image(image, angle):
    """Rotates an image around its center without cropping."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    abs_cos = abs(rot_mat[0,0]) 
    abs_sin = abs(rot_mat[0,1])
    bound_w = int(image.shape[1] * abs_cos + image.shape[0] * abs_sin)
    bound_h = int(image.shape[1] * abs_sin + image.shape[0] * abs_cos)

    rot_mat[0, 2] += bound_w/2 - image_center[0]
    rot_mat[1, 2] += bound_h/2 - image_center[1]

    result = cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    return result

def overlay_image_alpha(img, img_overlay, x, y):
    """Overlays a rotated RGBA image onto the background."""
    h, w, _ = img_overlay.shape
    
    y1, y2 = max(0, y), min(img.shape[0], y + h)
    x1, x2 = max(0, x), min(img.shape[1], x + w)
    
    overlay_y1, overlay_y2 = max(0, -y), min(h, img.shape[0] - y)
    overlay_x1, overlay_x2 = max(0, -x), min(w, img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or overlay_y1 >= overlay_y2 or overlay_x1 >= overlay_x2:
        return

    roi = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    alpha = img_overlay_crop[:, :, 3] / 255.0
    alpha_mask = np.dstack([alpha] * 3)

    roi_bg = roi.astype(float) * (1.0 - alpha_mask)
    roi_fg = img_overlay_crop[:, :, :3].astype(float) * alpha_mask
    
    blended_roi = cv2.add(roi_bg, roi_fg)
    img[y1:y2, x1:x2] = blended_roi.astype(np.uint8)


def process_video(cap, output_path, filter_image_path):
    """Processes video, applying a smoothed filter with accurate tracking."""
    # --- dlib Initialization ---
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except RuntimeError:
        print("Error: `shape_predictor_68_face_landmarks.dat` not found."); return

    # --- Load Filter Image & Create Transparency ---
    try:
        filter_img_bgr = cv2.imread(filter_image_path)
        if filter_img_bgr is None: raise FileNotFoundError
        original_filter_img = cv2.cvtColor(filter_img_bgr, cv2.COLOR_BGR2BGRA)
        white_pixels = np.all(filter_img_bgr >= [250, 250, 250], axis=-1)
        original_filter_img[white_pixels, 3] = 0
    except FileNotFoundError:
        print(f"Error: Filter image not found at '{filter_image_path}'"); return

    # --- Video Writer Setup ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # --- Initialize Stabilizers ---
    # We use separate stabilizers for position and size/angle for better results
    pos_stabilizer = Stabilizer(size=5)
    size_stabilizer = Stabilizer(size=5)
    angle_stabilizer = Stabilizer(size=7) # Angle can be more sensitive

    print("Processing video with smoothed tracking...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # We only track one face for simplicity
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)

            # --- Get raw measurements ---
            left_eye_raw = np.array([landmarks.part(36).x, landmarks.part(36).y])
            right_eye_raw = np.array([landmarks.part(45).x, landmarks.part(45).y])
            
            # --- Smooth the measurements ---
            eye_center = pos_stabilizer.update((left_eye_raw + right_eye_raw) / 2)
            
            eye_width_raw = np.linalg.norm(right_eye_raw - left_eye_raw)
            stable_eye_width = size_stabilizer.update(eye_width_raw)

            angle_raw = -np.degrees(math.atan2(right_eye_raw[1] - left_eye_raw[1], right_eye_raw[0] - left_eye_raw[0]))
            stable_angle = angle_stabilizer.update(angle_raw)
            
            # --- 2. Resize and Rotate the Filter using stable values ---
            filter_width = int(stable_eye_width * 1.5)
            h, w, _ = original_filter_img.shape
            filter_height = int(filter_width * (h / w))
            
            resized_filter = cv2.resize(original_filter_img, (filter_width, filter_height))
            rotated_filter = rotate_image(resized_filter, stable_angle)
            
            # --- 3. Calculate Position using stable values ---
            rh, rw, _ = rotated_filter.shape
            x = int(eye_center[0] - rw // 2)
            y = int(eye_center[1] - rh // 2)
            
            # --- 4. Overlay the final filter ---
            overlay_image_alpha(frame, rotated_filter, x, y)

        # --- Add Timestamp ---
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("Processing...", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

def main():
    """Main function to run the application."""
    choice = input("Choose input source (1 for webcam, 2 for video file): ")
    
    if choice == '1':
        cap = cv2.VideoCapture(0)
        output_path = 'output_webcam.mp4'
    elif choice == '2':
        video_path = upload_file("Select a Video File", (("MP4 files", "*.mp4"), ("AVI files", "*.avi")))
        if not video_path: return
        cap = cv2.VideoCapture(video_path)
        output_path = 'output_video_1.mp4'
    else:
        print("Invalid choice."); return

    if not cap.isOpened():
        print("Error: Could not open video source."); return
        
    filter_path = upload_file("Select a Filter Image", (("PNG files", "*.png"), ("JPEG files", "*.jpg")))
    if not filter_path:
        print("No filter image selected."); return

    process_video(cap, output_path, filter_path)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()