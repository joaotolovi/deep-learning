import cv2
xml = 'haarcascade_frontalface_alt2.xml'

faceClassifier=cv2.CascadeClassifier(xml)

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while not cv2.waitKey(20) & 0xFF == ord("q"):
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassifier.detectMultiScale(gray)

    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0))

    cv2.imshow('color', frame)
    cv2.imshow('gray', gray)
