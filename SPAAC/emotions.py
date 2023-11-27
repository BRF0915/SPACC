import cv2
import numpy as np
import dlib
from imutils import face_utils
import face_recognition
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

USE_WEBCAM = True # If false, loads video file source

# change
from collections import deque
import time
import pygame

def play_alert_sound():
    alert_sound = pygame.mixer.Sound("alert_sound.wav")
    alert_sound.play()

not_attentive_count = 0
not_attentive_queue = deque(maxlen=5)

not_face_count = 0
no_face_queue = deque(maxlen=5)

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
# emotion_labels = get_labels('fer2013')
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)

# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./test/testvdo.mp4') # Video file source

while cap.isOpened(): # True:
    ret, bgr_image = cap.read()

    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = detector(rgb_image)


    #change
    if not faces:
        face_label = 0
    else:
        face_label = 1

    no_face_queue.append(face_label)

    # 如果队列中的所有元素都是'not_attentive'，则增加计数器
    if all(em == 0 for em in no_face_queue):
        not_face_count += 1
    else:
        not_face_count = 0  # 重置计数器

    # 如果计数器达到5，触发异常警报
    if not_face_count >= 30:
        # 触发异常警报的代码，例如播放警报音乐或发送通知
        print('=============================')
        print('异常：无人值守')
        # play_alert_sound()

    # if not faces:
    #     print('No face detected')

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'ANGRY':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'DISGUST':
            color = emotion_probability * np.asarray((0, 250, 250))
        elif emotion_text == 'FEAR':
            color = emotion_probability * np.asarray((0, 0, 225))
        elif emotion_text == 'HAPPY':
            color = emotion_probability * np.asarray((0, 255, 0))
        elif emotion_text == 'SAD':
            color = emotion_probability * np.asarray((0, 0, 0))
        elif emotion_text == 'SURPRISE':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'NEUTRAL':
            color = emotion_probability * np.asarray((255, 0, 0))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()


        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)#emotion_mode
        draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

        #change
        not_attentive_queue.append(emotion_mode)

        # 如果队列中的所有元素都是'not_attentive'，则增加计数器
        if all(em == 'NOT ATTENTIVE' for em in not_attentive_queue):
            not_attentive_count += 1
        else:
            not_attentive_count = 0  # 重置计数器

        # 如果计数器达到5，触发异常警报
        if not_attentive_count >= 30:
            # 触发异常警报的代码，例如播放警报音乐或发送通知
            print('=============================')
            print('异常：操作不专注')
            # play_alert_sound()
        
        # print('EMOTION PROBABILITY -> ', emotion_probability)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
