#import an existing logistic regression model 
import joblib
import preprocess
from utils import open_dcm, divide_image
from extract_features import calculate_difference_old
import numpy as np
import pandas as pd




def logistic_construction(X):
    coefs = dict()
    coefs_csv = pd.read_csv("./model/model_coefs.csv")
    for index, row in coefs_csv.iterrows():
        coefs[row['Unnamed: 0']] = row['x']
    intercept = coefs['(Intercept)']
    features = [k for k in coefs.keys() if k != '(Intercept)']
    weights = np.array([coefs[f] for f in features])

    z = intercept + np.dot(X, weights)
    return 1 / (1 + np.exp(-z))


def preprocess_data(data = None, data_path = None, extract_window = False):
    if data is not None:
        images = []
        for i in range(len(data_path)):
            image = open_dcm(data_path[i])
            preprocessed = preprocess.preprocessing_image(image_path = data_path[i],\
                                    image=image, \
                                    gaussian_smoothing=True, \
                                    wavelets=True, extract_window_lesion=extract_window)
            images.append([image, preprocessed[1], preprocessed[2]])
        return images


def extract_features_from_data(data, data_path):
    if isinstance(data, list):
        preprocessed_images = preprocess_data(data=data, data_path=data_path, extract_window=True)
        features_list = []
        for image in preprocessed_images:
            right_image, left_image = divide_image(image=image[0])
            JSD, MAE, DSSIM = calculate_difference_old(left_half=left_image, right_half=right_image)
            right_CH, left_CH = divide_image(image=image[1])
            cH_JSD, _, cH_DSSIM = calculate_difference_old(left_half=left_CH, right_half=right_CH)
            right_CV, left_CV = divide_image(image=image[2])
            cV_JSD, _, cV_DSSIM = calculate_difference_old(left_half=left_CV, right_half=right_CV)
            features_list.append([cH_DSSIM, cH_JSD, cV_DSSIM, cV_JSD, JSD, DSSIM, MAE])
        print(features_list)
        return features_list



def classify_data(data_path):
    if isinstance(data_path, list):
        data = [open_dcm(path) for path in data_path]
    else:
        raise ValueError("data_path should be a list of file paths.")

    features = extract_features_from_data(data = data, data_path = data_path)
    prediction = logistic_construction(features)
    return prediction


if __name__ == "__main__":
    image_path = './data_example/MRIs/Patient_1/MRI_1/export_00062.DCM'
    
    prediction = classify_data(data_path=[image_path])
    print(prediction)