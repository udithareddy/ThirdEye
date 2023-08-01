from flask import Flask, render_template, Response
import cv2
import face_recognition
from deepface import DeepFace
import pyttsx3
import time

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('demo.html')


@app.route('/openeye')
def dummy():
    return render_template('index.html')


def gen_frames():
    capture = cv2.VideoCapture(0)
    engine = pyttsx3.init()
    while True:
        time.sleep(3)
        success, img = capture.read()
        if success:
            # grey = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            fcLoc = face_recognition.face_locations(img)[0]
            cv2.rectangle(img, (fcLoc[3], fcLoc[0]), (fcLoc[1], fcLoc[2]), (0, 255, 0), 2)
            result = DeepFace.analyze(img, actions=["emotion"])
            Emotion = result['dominant_emotion']
            cv2.rectangle(img, (fcLoc[3], fcLoc[0] - 35), (fcLoc[1], fcLoc[0]), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, Emotion, (fcLoc[3] + 6, fcLoc[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(img,Emotion,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            Feeling = f'The person seems to be {Emotion}'
            engine.say(Feeling)
            engine.runAndWait()
            ref, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
        else:
            reason = f'face cannot be detected, try again'
            engine.say(reason)
            engine.runAndWait()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/STOP')
def quit():
    return render_template('demo.html')


if __name__ == '__main__':
    app.run(debug=True)