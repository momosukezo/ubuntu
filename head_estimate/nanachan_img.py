import cv2
import dlib
import numpy as np
import pandas as pd
import time

#path
im = './kakukento.jpeg'
cascade = '../data/haarcascade_frontalface_default.xml'
predictor = '../data/shape_predictor_68_face_landmarks.dat'



#!/usr/bin/env python

# Read Image
img = cv2.imread(im)
size = img.shape
first = int(time.time() * 1000)


CASCADE = cv2.CascadeClassifier(cascade)
PREDICTOR = dlib.shape_predictor(predictor)
face_detector = dlib.get_frontal_face_detector()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
detected_faces = face_detector(gray, 2)
landmarks = []

for rect in detected_faces:
   cv2.rectangle(img, (rect.left(), rect.top()),(rect.right(), rect.bottom()), (0, 255, 0), 3)


   for rect in detected_faces:
      landmarks.append(np.array([[o.x,o.y] for o in PREDICTOR(img, rect).parts()]))


#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            landmarks[0][30],     # Nose tip
                            landmarks[0][8],     # Chin
                            landmarks[0][36],     # Left eye left corner
                            landmarks[0][45],     # Right eye right corne
                            landmarks[0][48],     # Left Mouth corner
                            landmarks[0][54]      # Right mouth corner
                        ], dtype="double")

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        
                        ])


# Camera internals

focal_length = size[1]
center = (size[1]/2, size[0]/2)
#print('center' + str(center))
#print('center[0]' + str(center[0]))
#print('center[1]' + str(center[1]))
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

#print ("Camera Matrix :\n {0}".format(camera_matrix));

dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
PnP = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

print ("Rotation Vector:\n {0}".format(rotation_vector))
print ("Translation Vector:\n {0}".format(translation_vector))


# Project a 3D point (0, 0, 1000.0) onto the image plane.
# z length 1000
# We use this to draw a line sticking out of the nose


(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 2000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
project = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

print('nose_end_point2D' + str(nose_end_point2D))

last = int(time.time() * 1000)
print('time' + str(last - first) + 'ms')
msSum = last - first
for p in image_points:
    cv2.circle(img, (int(p[0]),int(p[1])), 3, (0,0,255), -1)

p1 = ( int(image_points[0][0]), int(image_points[0][1]))
p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

cv2.line(img, p1, p2, (255,0,0), 2)




# Display image
cv2.imshow("Output", img);
K = cv2.waitKey(0)
if K == ord('q'):
   cv2.destroyAllWindows()
