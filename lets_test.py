import cv2 as cv2
import numpy as np
from keras.models import model_from_json
''

emotion_dict={0:"angry",1:"disgusted",2:"fearful",3:"happy",4:"neutral",5:"sad",6:"surprised"}

json_file = open('venv/emotion.json','r')
loaded_module_json = json_file.read()
json_file.close()
emotion=model_from_json(loaded_module_json)
emotion.load_weights("venv/emotion.h5")

cap = cv2.VideoCapture(0)


while True:
    cap.set(cv2.CAP_PROP_FPS, 60)
    ret, frame = cap.read()
    print(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    frame = cv2.resize(frame, (600,600 ))
    if not ret:
        break
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
