import cv2
import dlib
import numpy as np
import time
import pandas as pd

# path
video = '../zemi/MI09_56m_rhythm.avi'
cascade = '../data/haarcascade_frontalface_default.xml'
predictor = '../data/shape_predictor_68_face_landmarks.dat'

# import data
cap = cv2.VideoCapture(video)
CASCADE = cv2.CascadeClassifier(cascade)
PREDICTOR = dlib.shape_predictor(predictor)
face_detector = dlib.get_frontal_face_detector()
msSum = 0
msSum_2 = 0
toki = []
tukene = []
houko = []
tukene_2 = []
houko_2 = []

while(cap.isOpened()):
   ret, frame = cap.read()
   if ret == False:
      break
   size = frame.shape
   first = int(time.time() * 1000)
   

   gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   detected_faces = face_detector(gray, 1) # scalesize
   landmarks = []
   
   for rect in detected_faces:
      landmarks.append(np.array([[o.x,o.y] for o in PREDICTOR(gray, rect).parts()]))

   if len(landmarks) == 1:
      image_points1 = np.array([
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
      camera_matrix = np.array(
                          [[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype = "double"
                         )

      #print ("Camera Matrix :\n {0}".format(camera_matrix));

      dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
      (success1, rotation_vector1, translation_vector1) = cv2.solvePnP(model_points, image_points1, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

      #print ("Rotation Vector:\n {0}".format(rotation_vector))
      #print ("Translation Vector:\n {0}".format(translation_vector))


# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose


      (nose_end_point2D1, jacobian1) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector1, translation_vector1, camera_matrix, dist_coeffs)

      #print(image_points)
      for p in image_points1:
         cv2.circle(frame, (int(p[0]),int(p[1])), 3, (0,0,255), -1)


      p11 = ( int(image_points1[0][0]), int(image_points1[0][1]))
      p21 = ( int(nose_end_point2D1[0][0][0]), int(nose_end_point2D1[0][0][1]))

      cv2.line(frame, p11, p21, (255,0,0), 2)

   elif len(landmarks) == 2:
      image_points1 = np.array([
                            landmarks[0][30],     # Nose tip
                            landmarks[0][8],     # Chin
                            landmarks[0][36],     # Left eye left corner
                            landmarks[0][45],     # Right eye right corne
                            landmarks[0][48],     # Left Mouth corner
                            landmarks[0][54]      # Right mouth corner
                                ], dtype="double")
      image_points2 = np.array([
                            landmarks[1][30],     # Nose tip
                            landmarks[1][8],     # Chin
                            landmarks[1][36],     # Left eye left corner
                            landmarks[1][45],     # Right eye right corne
                            landmarks[1][48],     # Left Mouth corner
                            landmarks[1][54]      # Right mouth corner
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
      camera_matrix = np.array(
                          [[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype = "double"
                         )

      #print ("Camera Matrix :\n {0}".format(camera_matrix));

      dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
      (success1, rotation_vector1, translation_vector1) = cv2.solvePnP(model_points, image_points1, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
      (success2, rotation_vector2, translation_vector2) = cv2.solvePnP(model_points, image_points2, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

      #print ("Rotation Vector:\n {0}".format(rotation_vector))
      #print ("Translation Vector:\n {0}".format(translation_vector))


# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose


      (nose_end_point2D1, jacobian1) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector1, translation_vector1, camera_matrix, dist_coeffs)
      (nose_end_point2D2, jacobian2) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector2, translation_vector2, camera_matrix, dist_coeffs)

      #print(image_points)
      for p in image_points1:
          cv2.circle(frame, (int(p[0]),int(p[1])), 3, (0,0,255), -1)
      for q in image_points2:
          cv2.circle(frame, (int(q[0]),int(q[1])), 3, (0,0,255), -1)

      last_2 = int(time.time())
      msSum_2 = msSum_2 + (last_2 - first)
      p11 = ( int(image_points1[0][0]), int(image_points1[0][1]))
      p21 = ( int(nose_end_point2D1[0][0][0]), int(nose_end_point2D1[0][0][1]))
      p12 = ( int(image_points2[0][0]), int(image_points2[0][1]))
      p22 = ( int(nose_end_point2D2[0][0][0]), int(nose_end_point2D2[0][0][1]))
      '''
      TIME.append(msSum)
      nosePoint_1.append(p11)
      headEst_1.append(p21)
      nosePoint_2.append(p12)
      headEst_2.append(p22)
      '''
      df = pd.DataFrame([toki,tukene,houko,tukene_2,houko_2],
                         index=['Time','nosePoint_1','headEstimate_1','nosePoint_2','headEstimate_2'])
      dfr =  df.T
      dfr.to_csv('test.csv')


      cv2.line(frame, p11, p21, (255,0,0), 2)
      cv2.line(frame, p12, p22, (255,0,0), 2)
     
      
# time_edition
   last = int(time.time() * 1000)
   msSum = msSum + (last - first)
   #print('time >> ' + str(msSum) + 'ms')

# Display image
   cv2.imshow("frmame", frame)
   K = cv2.waitKey(1)
   if K == ord('q'):
      break


cap.release()
cv2.destroyAllwindows()
