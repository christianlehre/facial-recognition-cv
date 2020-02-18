import numpy as np 
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

def train(train_dir,model_save_path=None,n_neighbors=None,knn_algo='ball_tree'):
    """
    Train a KNN classifier and save it for later use/predictions
    """
    X = []
    y = []

    # loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        # only consider folders
        if not os.path.isdir(os.path.join(train_dir,class_dir)):
            continue

        # loop though each image for current person (class_dir)
        for img_path in image_files_in_folder(os.path.join(train_dir,class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) != 1:
                print("[INFO] Image {} not suited for training: {}".format(img_path,"No face" if len(face_locations) < 1 else "Multiple faces"))
                continue
            else:
                face_encodings = face_recognition.face_encodings(image,face_locations)[0]
                X.append(face_encodings)
                y.append(class_dir)
        print("[INFO] done with {}".format(class_dir))

        if n_neighbors is None:
            n_neighbors = int(round(np.sqrt(len(X))))
            print("Number of neighbors chosen automatically: {}".format(n_neighbors))

        # create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=knn_algo,weights='distance')
        knn_clf.fit(X,y)

        # save the trained classifier to later predictions
        if model_save_path is not None:
            with open(model_save_path,'wb') as f:
                pickle.dump(knn_clf,f)
        
    return knn_clf

def predict(X_img_path,model_path,distance_threshold=0.6):
    """
    input: 
        X_img_path: path to image in which faces are to be recognized by the knn classifier
        model_path: path to a pickled knn classifier
        distance_threshold: threshold value for classification

    output: list of names and location of faces [(name,locations), ...]
    """

    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    if len(X_face_locations) == 0:
        print("No faces in input image")
        return []

    face_encodings = face_recognition.face_encodings(X_img,X_face_locations)

    # load knn model and find best matches for the test image
    with open(model_path,'rb') as f:
        knn_clf = pickle.load(f)

    closest_distances = knn_clf.kneighbors(face_encodings,n_neighbors=1)
    matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    return [(pred,loc) if rec else ("Unknown",loc) for pred,loc,rec in zip(knn_clf.predict(face_encodings),X_face_locations,matches)]

def show_prediction(img_path,predictions):

    pil_image = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(pil_image)

    for name,(top,right,bottom,left)in predictions:
        draw.rectangle(((left,top),(right,bottom)),outline=(255,0,0))
        name = name.encode("UTF-8")
        _,text_height = draw.textsize(name)
        draw.rectangle(((left,bottom-text_height-10),(right,bottom)),fill=(255,0,0),outline=(255,0,0))
        draw.text((left+6,bottom-text_height-5),name,fill = (255,255,255,255))

    del draw
    pil_image.show()

if __name__ == "__main__":
    # Step 1: train model (comment out if already trained)
    """
    train_dir = "./facial_recognition_ML/train_dir"
    print("Training model ...")
    classifier = train(train_dir,model_save_path="./facial_recognition_ML/trained_knn_model.clf",n_neighbors=2)
    print("... Training complete!")
    """
    test_img_path = "./facial_recognition_ML/bill_gates_unknown.jpg"
    model_path = "./facial_recognition_ML/trained_knn_model.clf"

    predictions = predict(test_img_path,model_path)

    show_prediction(test_img_path,predictions)


