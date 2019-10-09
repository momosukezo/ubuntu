import numpy as np
import pandas as pd
import cv2
import dlib
import time

# path
video = '../zemi/MI09_89m_rhythm.avi'
cascade = '../data/haarcascade_frontalface_default.xml'
predictor = '../data/shape_predictor_68_face_landmarks.dat'

# import data
cap = cv2.VideoCapture(video)
CASCADE = cv2.CascadeClassifier(cascade)
PREDICTOR = dlib.shape_predictor(predictor)
face_detector = dlib.get_frontal_face_detector()


# Variable
'''
msSum = 0
TIME = []
Nose_Point_1 = []
Head_Est_1 = []
Nose_Point_2 = []
Head_Est_2 = []

NaN = [0]
data = np.zeros((1000,4))
mat = data[:,:]
df = pd.DataFrame(data)
df.to_csv('test.csv')
'''
df = pd.DataFrame([],index=[],columns=['noface','1_rotation_x','1_rotation_y','1_rotation_z','1_translation_x','1_translation_y','1_translation_z','2_rotation_x','2_rotation_y','2_rotation_z','2_translation_x','2_translation_y','2_translation_z'])

# Video_ON
while(cap.isOpened()):
   ret, frame = cap.read()

   if ret == False:
      break

   size = frame.shape

   first = int(time.time() * 1000)  # start_time
   
   gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   detected_faces = face_detector(gray, 1) # ←Lebel Of detect
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

#      print ("Rotation Vector:\n {0}".format(rotation_vector1))
#      print(rotation_vector1[0])
      
      #print ("Translation Vector:\n {0}".format(translation_vector))
      
      df_1 = pd.DataFrame([rotation_vector1[0],rotation_vector1[1],rotation_vector1[2],translation_vector1[0],translation_vector1[1],translation_vector1[2]],index=['1_rotation_x','1_rotation_y','1_rotation_z','1_translation_x','1_translation_y','1_translation_z'])

      
      df_1 = df_1.T
      df = df.append(df_1)


# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose


      (nose_end_point2D1, jacobian1) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector1, translation_vector1, camera_matrix, dist_coeffs)

      #print(image_points)
      for p in image_points1:
         cv2.circle(frame, (int(p[0]),int(p[1])), 3, (0,0,255), -1)


      p11 = ( int(image_points1[0][0]), int(image_points1[0][1]))
      p21 = ( int(nose_end_point2D1[0][0][0]), int(nose_end_point2D1[0][0][1]))

      '''
      TIME.append(msSum)
      Nose_Point_1.append(p11)
      Head_Est_1.append(p21)
      #Nose_Point_2.append(p12)
      #Head_Est_2.append(p22)
      '''

      cv2.line(frame, p11, p21, (255,0,0), 2)
      

      #print(df)
      #dfr =  df.T
      #dfr.to_csv('test.csv')

   elif len(landmarks) == 2:
      image_points1 = np.array([
                            landmarks[0][30],     # Nose tip
                            landmarks[0][8],      # Chin
                            landmarks[0][36],     # Left eye left corner
                            landmarks[0][45],     # Right eye right corne
                            landmarks[0][48],     # Left Mouth corner
                            landmarks[0][54]      # Right mouth corner
                                ], dtype="double")
      image_points2 = np.array([
                            landmarks[1][30],     # Nose tip
                            landmarks[1][8],      # Chin
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


      dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
      (success1, rotation_vector1, translation_vector1) = cv2.solvePnP(model_points, image_points1, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
      (success2, rotation_vector2, translation_vector2) = cv2.solvePnP(model_points, image_points2, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)





      (nose_end_point2D1, jacobian1) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector1, translation_vector1, camera_matrix, dist_coeffs)
      (nose_end_point2D2, jacobian2) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector2, translation_vector2, camera_matrix, dist_coeffs)

      for p in image_points1:
          cv2.circle(frame, (int(p[0]),int(p[1])), 3, (0,0,255), -1)
      for q in image_points2:
          cv2.circle(frame, (int(q[0]),int(q[1])), 3, (0,0,255), -1)

      last = int(time.time() * 1000) # end_time
      #msSum = msSum + (last - first)

      # Nose_point
      p11 = ( int(image_points1[0][0]), int(image_points1[0][1]))
      p21 = ( int(nose_end_point2D1[0][0][0]), int(nose_end_point2D1[0][0][1]))
      p12 = ( int(image_points2[0][0]), int(image_points2[0][1]))
      p22 = ( int(nose_end_point2D2[0][0][0]), int(nose_end_point2D2[0][0][1]))
      '''

      TIME.append(msSum)
      Nose_Point_1.append(p11)
      Head_Est_1.append(p21)
      Nose_Point_2.append(p12)
      Head_Est_2.append(p22)
      '''

    
      df_2 = pd.DataFrame([rotation_vector1[0],rotation_vector1[1],rotation_vector1[2],translation_vector1[0],translation_vector1[1],translation_vector1[2],rotation_vector2[0],rotation_vector2[1],rotation_vector2[2],translation_vector2[0],translation_vector2[1],translation_vector2[2]],index=['1_rotation_x','1_rotation_y','1_rotation_z','1_translation_x','1_translation_y','1_translation_z','2_rotation_x','2_rotation_y','2_rotation_z','2_translation_x','2_translation_y','2_translation_z'])

      
      df_2 = df_2.T
      df = df.append(df_2)
      #df_2 = df_2
      df = df.append(df_2)

   #   dfr =  df.T
   #   dfr.to_csv('test.csv')


      cv2.line(frame, p11, p21, (255,0,0), 2)
      cv2.line(frame, p12, p22, (255,0,0), 2)
   
   else:
      df_3 = pd.DataFrame([0],index=['noface'])
      df_3 = df_3.T
      df = df.append(df_3)
   #df = pd.DataFrame([Nose_Point_1,Head_Est_1,Nose_Point_2,Head_Est_2],
   #index=['NosePoint_1','NoseEnd_1','NosePoint_2','NoseEnd_2'])
   #dfr =  df.T
   #dfr.to_csv('test.csv')
   cv2.imshow("frmame", frame)
   N = cv2.waitKey(1)
   if N == ord('s'):
      df.to_csv('test.csv')
      break
   '''
   K = cv2.waitKey(1)
   if K == ord('q'):
      cv2.AllWindows()
   '''
cap.release()
cv2.AllWindows()
