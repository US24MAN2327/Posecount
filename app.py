import streamlit as st


import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO
import cv2
import mediapipe as mp
# Initialize YOLOv8 model for person detection
yolo_model = YOLO('yolov8n.pt')  # Use yolov8n.pt (nano) for small model; switch to other YOLOv8 models if needed

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Set up Streamlit layout
st.set_page_config(page_title="Multi-Person Pose Estimation", page_icon="🤸", layout="wide")
st.title("Multi-Person Pose Estimation with YOLOv8 and MediaPipe")
st.write("Upload an image or video to detect multiple persons and estimate their poses.")

# YOLO Function to detect people
def detect_people_yolov8(image):
    # Perform inference using YOLOv8
    results = yolo_model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes as numpy array
    return detections

# Pose Estimation function for detected people
def estimate_pose_on_person(image, bbox):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    person_roi = image[y1:y2, x1:x2]
    
    # Convert the image to RGB for MediaPipe Pose processing
    person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
    result = pose.process(person_rgb)
    
    # If pose landmarks are detected, draw them on the ROI
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(person_roi, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return image

# Function to detect and estimate poses on an image, and count people
def process_image(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect people using YOLOv8
    detections = detect_people_yolov8(image_bgr)
    person_count = len(detections)  # Count the number of people detected
    
    # Apply pose estimation for each detected person
    for bbox in detections:
        image_bgr = estimate_pose_on_person(image_bgr, bbox)

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), person_count

# Function to process video and count people in each frame
def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    frame_count = 0
    total_person_count = 0  # To keep track of total persons detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and process each frame for pose estimation
        processed_frame, person_count = process_image(frame)
        total_person_count += person_count
        out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

        frame_count += 1
        progress.progress(int(frame_count / total_frames * 100))

    cap.release()
    out.release()

    avg_person_count = total_person_count / frame_count if frame_count > 0 else 0
    return avg_person_count

# Streamlit Upload Options
upload_type = st.selectbox("Choose upload type", ("Image", "Video"))

if upload_type == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
        
        # Process the image
        processed_image, person_count = process_image(image)
        
        # Display the processed image and the number of detected persons
        st.image(processed_image, caption=f"Processed Image - {person_count} person(s) detected", use_column_width=True)

        # Download button for processed image
        result_image = Image.fromarray(processed_image)
        st.download_button("Download Processed Image", result_image.tobytes(), "processed_image.png", "image/png")

elif upload_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        # Process video and get average person count
        output_video_path = "output_video.mp4"
        avg_person_count = process_video(tfile.name, output_video_path)

        # Display processed video and average person count
        st.video(output_video_path)
        st.write(f"Average number of persons detected per frame: {avg_person_count:.2f}")

        # Download button for processed video
        with open(output_video_path, 'rb') as f:
            st.download_button("Download Processed Video", f, "processed_video.mp4")
