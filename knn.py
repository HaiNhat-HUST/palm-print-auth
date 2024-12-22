import math
from sklearn import neighbors
import os
import pickle
import cv2
import gabor
from tqdm import tqdm
import numpy as np

def train(train_dir, model_save_path=None,  n_neighbors=None, knn_algo='ball_tree', verbose=False):
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
            
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
        
    X = np.array(X)
    
    """
    nsampels, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    """
    

    # Create and trainn KNN
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    
    return knn_clf
        

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
            
    faces_encodings = gabor.process(X_img_path)

    # print(faces_encodings.shape)
    faces_encodings = np.array(faces_encodings)
    faces_encodings = faces_encodings.reshape(1, -1)

    # print(knn_clf.predict_proba(faces_encodings))
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    return knn_clf.predict(faces_encodings)
    

if __name__ == "__main__":
    
    """
    print("Training KNN...")
    classifier = train("./train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")
    
    """
    print("Testing KNN...")
    correct_prediction = 0
    print(len(os.listdir('./test')))
    for image_dir in os.listdir('./test'):
        full_file_path = './test/' + image_dir
        # print(full_file_path)
        
        # print("Looking for palm print in {}".format(image_dir))
        
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")
        # print("Real prediction: ", predictions)
        
        
        y_test = str(predictions[0])
        y_pred = image_dir[:4]
        if y_test == y_pred:
            correct_prediction += 1
        else:
            print("Looking for palm print in {}".format(image_dir))
            print("Real prediction: ", predictions)

        
        
    accuracy = correct_prediction / len(os.listdir('./test'))
    print(f"Accuracy: {accuracy}")
    