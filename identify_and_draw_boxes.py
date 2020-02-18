from PIL import Image,ImageDraw
import face_recognition
import numpy as np
import glob
import os


def learn_known_faces(parent_folder):
    known_face_encodings = []
    known_face_names = []

    for filepath in glob.iglob(parent_folder):

        path_without_extension = os.path.splitext(filepath)[0]
        name = path_without_extension.split("/")[-1]

        known_image = face_recognition.load_image_file(filepath)
        known_encoding = face_recognition.face_encodings(known_image)[0]

        known_face_names.append(name)
        known_face_encodings.append(known_encoding)

    return known_face_names,known_face_encodings


def find_faces_and_draw_boxes(path):
    """
    find all faces in the image specified by path and draw rectangles around them
    """
    image = face_recognition.load_image_file(path)

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image,face_locations)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for (top,right,bottom,left) in face_locations:
        draw.rectangle(((left,top),(right,bottom)),outline = (255,0,0))

    del draw
    return pil_image, face_locations,face_encodings

def identify_faces_and_draw_boxes(path,known_face_names,known_face_encodings):
    #load image with unknown face(s)
    unknown_image = face_recognition.load_image_file(path)

    #find location of faces and encodings in unknown image
    face_locations = face_recognition.face_locations(unknown_image,number_of_times_to_upsample=0,model='cnn')
    face_encodings = face_recognition.face_encodings(unknown_image,face_locations)

    #convert unknown image to a PIL-format image to be able to draw on top of it
    pil_image = Image.fromarray(unknown_image)
    #create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    #loop though faces in unknown image, match with faces in known images and draw boxes with labels
    for (top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):
        #set default name to unknown
        name = "Unknown"

        #match known faces to faces in unknwon image
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)

        # classify faces regarding to a minimal "face distance" (similiarity matching)
        face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        #draw rectangle around face
        draw.rectangle(((left,top),(right,bottom)),outline = (255,0,0))

        #draw label with name below face
        _,text_height = draw.textsize(name)
        draw.rectangle(((left,bottom-text_height-10),(right,bottom)),fill = (255,0,0),outline = (255,0,0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    del draw

    return pil_image

if __name__ == "__main__":
    path_to_known = "./known_people/*.jpg"
    path_to_unknown = "./unknown_pictures/the_four_big.jpg"

    #names,encodings = learn_known_faces(path_to_known)
    #image = identify_faces_and_draw_boxes(path_to_unknown,names,encodings)
    #image.show()
    image,_,_ = find_faces_and_draw_boxes(path_to_unknown)
    image.show()
    #if len(names) > 0:
    #    image.save("./identified/the_four_big.jpg")




