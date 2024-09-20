import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    print('hand landmarker result: {}'.format(result))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands = 1,#only detect one hand for now
    min_hand_detection_confidence = 0.3, #lower accuracy for easier success
    min_hand_presence_confidence = 0.3,
    min_tracking_confidence = 0.5,
    result_callback=print_result)


#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detected_image):
    hand_landmarks_list = detected_image.hand_landmarks
    handedness_list = detected_image.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

    return annotated_image



#time stamper

def main():
    cap = cv.VideoCapture(0) #turn on
#    lm_visualizer = Landmark_Visualizer()
    #start OpenCV's timer to get relative timestamp
    #set window dimensions
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Cannot open camera.")
        exit() 

    #reading data
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            #current time
            ret, frame = cap.read()
            #open camera, read frame into numpy array, changing rgb to bgr and back bc openCV and mediapipe use opposites
            frame = cv.flip(frame,1) #opencv window outputs BGR, mediapipe reads RGB
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)
            #monotonically increasing timestamp
    #           print(mp.Timestamp.is_allowed_in_stream(int(time.time()*1000)))
            cap_time = int(time.time()*1000)
            #use visualizer to detect and visualize landmarks
            #detect_async second pos arg does not accept functions or int, using openCV's tick since camera is open
            #somehow setting cap_time to int(time.time()*1000 works)??? sec pos kinda fixed
            detected_image = landmarker.detect_async(mp_image,cap_time)
            if detected_image is None:
                annotated_image = frame
            else:
                rgb_image = frame
                annotated_image = draw_landmarks_on_image(rgb_image, detected_image)

            cv.imshow('Webcam',annotated_image)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    with HandLandmarker.create_from_options(options) as landmarker:
        main()




    



