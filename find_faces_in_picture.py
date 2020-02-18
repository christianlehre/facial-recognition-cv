from PIL import Image
import face_recognition

path = './unknown_pictures/musk_others.jpg'

# open image that are undergoing facial recognition
im = Image.open(path)
im.show()

#load image (.jpg) into numpy array
image = face_recognition.load_image_file(path)

# find locations (top,right,bottom,left)
face_locations = face_recognition.face_locations(image)

print("There are {} face(s) in the photograph!".format(len(face_locations)))

for loc in face_locations:
    t,r,b,l = loc

    face_image = image[t:b,l:r]
    pil_image = Image.fromarray(face_image)
    pil_image.show()