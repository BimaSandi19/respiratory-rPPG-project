import cv2

def capture_video():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Could not open video device")

    return cap

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise Exception("Could not read frame from video device")
    
    return frame

def convert_frame_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def release_video_capture(cap):
    cap.release()
    cv2.destroyAllWindows()