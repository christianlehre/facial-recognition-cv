import face_recognition
from sklearn import svm
import os
from PIL import Image, ImageDraw

# data consists of face encodings and name labels
encodings = []
names = []

path_to_training_data = "./train_dir/"
train_dir = os.listdir(path_to_training_data)

for person in train_dir:
    # ignore .DS_Store etc
    if person.startswith('.'):
        continue

    # folder containing images of one person
    person_pics = os.listdir(path_to_training_data + person)
    # Loop through all images in a persons folder, for all persons in the training folder
    for person_img in person_pics:
        # ignore .DS_Store
        if person_img.startswith('.'):
            continue

        face = face_recognition.load_image_file(path_to_training_data +person + "/" + person_img)
        face_locations = face_recognition.face_locations(face)

        if len(face_locations) != 1:
            print("[INFO] " + person + "/" + person_img + " contains none or more than one face, cannot be used for training")
            continue
        else:
            face_encoding = face_recognition.face_encodings(face)[0]
            encodings.append(face_encoding)
            names.append(person)
    print("[INFO] done with {}".format(person))

# create and train the classifier
clf = svm.SVC(gamma = 'scale')
clf.fit(encodings,names)

test_image = face_recognition.load_image_file('./test_image.jpg')

pil_image = Image.fromarray(test_image)
draw = ImageDraw.Draw(pil_image)

face_locations = face_recognition.face_locations(test_image)
num_faces = len(face_locations)
print("Number of faces detected: ", num_faces)

# Predict all the faces in the test image, using the classifier
print("Found:")
for i,(top,right,bottom,left) in zip(range(num_faces),face_locations):
    test_image_encoding = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_encoding])
    _,text_height = draw.textsize(*name)
    draw.rectangle(((left,top),(right,bottom)),outline = (255,0,0))
    draw.rectangle(((left,bottom-text_height-10),(right,bottom)),fill = (255,0,0),outline = (255,0,0))
    draw.text((left + 6, bottom - text_height - 5), str(*name), fill=(255, 255, 255, 255))
    print(*name)


del draw
pil_image.show()
pil_image.save("./classified/MuskGateJobs.jpg")