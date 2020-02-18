from PIL import Image
import face_recognition

# open image that are undergoing facial recognition
im = Image.open('./unknown_pictures/musk_others.jpg')
im.show()

#load image using facial recognition
image = face_recognition.load_image_file("./unknown_pictures/musk_others.jpg")


face_locations = face_recognition.face_locations(image,number_of_times_to_upsample=0,model="cnn")


print("There are {} face(s) in the photo!".format(len(face_locations)))

for loc in face_locations:
    t,r,b,l = loc

    face_image = image[t:b,l:r]
    pil_image = Image.fromarray(face_image)
    pil_image.show()