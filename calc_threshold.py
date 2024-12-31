import os
import pickle
import gabor
import numpy as np

def find_person_and_calculate_distance(roi_image_path, model_path="trained_rf_model.clf", train_dir="./train_ver2"):
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)


    roi_features = gabor.process(roi_image_path)
    roi_features = np.array(roi_features).reshape(1, -1)

    # Predict the person using the Random Forest model
    predicted_person = rf_model.predict(roi_features)[0]

    # Calculate the minimum distance between the ROI image and images of the predicted person in the training set
    min_distance = float('inf')
    for img_path in os.listdir(f"{train_dir}/{predicted_person}"):
        train_image_path = f"{train_dir}/{predicted_person}/{img_path}"
        train_features = gabor.process(train_image_path)
        distance = np.linalg.norm(roi_features - train_features)
        if distance < min_distance:
            min_distance = distance

    return predicted_person, min_distance
