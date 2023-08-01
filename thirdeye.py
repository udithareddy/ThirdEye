import cv2
from deepface import DeepFace
import face_recognition
import pyttsx3


def speak(text):
    engine.say(text)
    engine.runAndWait()


engine = pyttsx3.init()
capture = cv2.VideoCapture(1)
if not capture.isOpened():
    capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise IOError("Cannot open webcam")

while True:
    success, img = capture.read()
    result = DeepFace.analyze(img, actions=["emotion"])
    Emotion = result['dominant_emotion']
    if success:
        fcLoc = face_recognition.face_locations(img)[0]

        cv2.rectangle(img, (fcLoc[3], fcLoc[0]), (fcLoc[1], fcLoc[2]), (0, 255, 0), 2)
        cv2.rectangle(img, (fcLoc[3], fcLoc[0] - 35), (fcLoc[1], fcLoc[0]), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, Emotion, (fcLoc[3] + 6, fcLoc[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        Feeling = f'The person seems to be {Emotion}'
        speak(Feeling)
        cv2.imshow('Live Video', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        print("We've faced a trouble with you Cam, Please check your Cam...")
        break

capture.release()
cv2.destroyAllWindows()