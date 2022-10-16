import cv2
import numpy as np
import face_recognition

#Main Image
imgVuyani = face_recognition.load_image_file('01.jpg')
imgVuyani = cv2.cvtColor(imgVuyani,cv2.COLOR_BGR2RGB)
#Test Image
imgVTest = face_recognition.load_image_file('03.jpg')
imgVTest = cv2.cvtColor(imgVTest,cv2.COLOR_BGR2RGB)

#finding face and encodings
#because we are sending in a single image we get the first element[0]
faceLoc = face_recognition.face_locations(imgVuyani)[0]
encodeVuyani = face_recognition.face_encodings(imgVuyani)[0]

faceLocTest = face_recognition.face_locations(imgVTest)[0]
encodeVTest = face_recognition.face_encodings(imgVTest)[0]

#Drawing Rectangle
start_point = (faceLoc[3], faceLoc[0], )
end_point = (faceLoc[1], faceLoc[2])
start_pointTest = (faceLocTest[3], faceLocTest[0], )
end_pointTest = (faceLocTest[1], faceLocTest[2])
rec_color = (255, 0, 0)
rec_thickness = 2
cv2.rectangle(imgVuyani,start_point,end_point,rec_color,rec_thickness)
cv2.rectangle(imgVTest,start_pointTest,end_pointTest,rec_color,rec_thickness)

results = face_recognition.compare_faces([encodeVuyani], encodeVTest)
faceDis = face_recognition.face_distance([encodeVuyani], encodeVTest)
print(results,faceDis)
cv2.putText(imgVTest, f'{results} {round(faceDis[0], 2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('Vuyani', imgVuyani)
cv2.imshow('Vuyani Test', imgVTest)
cv2.waitKey(0)