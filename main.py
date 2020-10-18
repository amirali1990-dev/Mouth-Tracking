import numpy as np
import dlib
import cv2
import os
import imutils
import face_recognition
from imutils import face_utils
from collections import OrderedDict

# face_detector = dlib.get_frontal_face_detector()   
predictor = dlib.shape_predictor("detector\\shape_predictor_68_face_landmarks.dat")

prototxtPath = "face_detector\\deploy.prototxt"
weightsPath = "face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

FACIAL_LANDMARKS_IDXS = OrderedDict([("mouth", (48, 68))])

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    overlay = image.copy()
    output = image.copy()
    if colors is None:
        colors = [(0, 0, 255)]
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]
        if name == "mouth":
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], 0, colors[i], 3)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

cv2.namedWindow("Mouth Tracking")
cap = cv2.VideoCapture(0)
    
if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

while ret:

    ret, frame = cap.read()

    frame = imutils.resize(frame, width=600)

    (h, w) = frame.shape[:2]

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    blob = cv2.dnn.blobFromImage(frame, 1.0, (190, 190),
        (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    face_rects = box.astype("int")

    if max(face_rects)<599:
        cv2.rectangle(frame, (face_rects[0], face_rects[1]), (face_rects[2], face_rects[3]), (0,255,0), 2)

    rect = dlib.rectangle(left=np.int(face_rects[0]), top=np.int(face_rects[1]), right=np.int(face_rects[2]), bottom=np.int(face_rects[3]))
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    output = visualize_facial_landmarks(frame, shape)
    cv2.imshow("Mouth Tracking", output)
    if cv2.waitKey(1) == 27:
        break
