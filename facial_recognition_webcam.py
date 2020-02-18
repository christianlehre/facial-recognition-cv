import face_recognition
import cv2
import numpy as np 

video_capture = cv2.VideoCapture(0)

christian_image = face_recognition.load_image_file('./known_people/Christian.jpg')
christian_encoding = face_recognition.face_encodings(christian_image)[0]

lise_image = face_recognition.load_image_file('./known_people/Lise.jpg')
lise_encoding = face_recognition.face_encodings(lise_image)[0]

mari_image = face_recognition.load_image_file('./known_people/Mari.jpg')
mari_encoding = face_recognition.face_encodings(mari_image)[0]

maria_image = face_recognition.load_image_file('./known_people/Maria.jpg')
maria_encoding = face_recognition.face_encodings(maria_image)[0]

known_face_encodings = [christian_encoding,lise_encoding,mari_encoding,maria_encoding]
known_face_names = ["Christian","Lise","Mari","Maria"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    #grab a single frame 
    ret,frame = video_capture.read()

    small_frame = cv2.resize(frame,(0,0), fx = 0.2, fy = 0.2)

    #convert from openCV's BGR to standard RGB
    rgb_small_frame = small_frame[:,:,::-1]

    # only process every other frame
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
            name = "Unknown" #default

            face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame



    # loop through faces in frame and draw boxes
    for (top,right,bottom,left),name in zip(face_locations,face_names):
        top *= 5
        right *= 5
        bottom *= 5
        left *=5

        # box around face
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)

        # label with name below face
        cv2.rectangle(frame,(left,bottom-5),(right,bottom),(0,0,255),cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)


    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
