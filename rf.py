import math
from sklearn import neighbors
import os
import pickle
import cv2
import gabor
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def train(train_dir, model_save_path=None):
    X = []
    y = []
    
    for class_dir in tqdm(os.listdir(train_dir)):

        for img_path in os.listdir(train_dir + "/" + class_dir):
            # image = cv2.imread(train_dir + "/" + class_dir + "/" + img_path)
            image = train_dir + "/" + class_dir + "/" + img_path
            H = gabor.process(image)
            
            # Add face encoding for current image to training set
            X.append(H)
            y.append(class_dir)
    
    X = np.array(X)
    
    """
    nsampels, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    """
    
    rf_clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)

    rf_clf.fit(X, y)
    
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(rf_clf, f)

    return rf_clf
        

def predict(X_img_path, clf=None, model_path=None):
    if clf is None:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
            
    faces_encodings = gabor.process(X_img_path)

    # print(faces_encodings.shape)
    faces_encodings = np.array(faces_encodings)
    faces_encodings = faces_encodings.reshape(1, -1)

    # print(svm_clf.predict(faces_encodings))
    
    return clf.predict(faces_encodings)
    

if __name__ == "__main__":
        
    print("Training random forest...")
    classifier = train("./train", model_save_path="trained_rf_model.clf")
    print("Training complete!")
    

    print("Testing random forest...")
    correct_prediction = 0
    print(len(os.listdir('./test')))
    for image_dir in tqdm(os.listdir('./test')):
        full_file_path = './test/' + image_dir
        # print(full_file_path)
        
        # print("Looking for palm print in {}".format(image_dir))
        
        predictions = predict(full_file_path, model_path="trained_rf_model.clf")
        # print("Real prediction: ", predictions)
        
        y_test = str(predictions[0])
        y_pred = image_dir[:4]
        if y_test == y_pred:
            correct_prediction += 1
        
        
    accuracy = correct_prediction / len(os.listdir('./test'))
    print(f"Accuracy: {accuracy}")
    