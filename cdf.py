from utils import black_white_conversion
import cv2
import numpy as np

def calc_hist(image):
    hist = [0] * 256
    for i in range(image.shape[0]): 
        for j in range(image.shape[1]): 
            hist[image[i, j]] += 1 
    return np.array(hist)

def calculate_cdf(hist):
    cdf = np.array([])
    for i in range(len(hist)):
        if i == 0:
            cdf = np.append(cdf, hist[i])
        else:
            cdf = np.append(cdf, cdf[i-1] + hist[i])
    return cdf

def get_threshold(cdf, v):
    threshold = 0
    for i in range(len(cdf)):
        if cdf[i] >= v:
            threshold = i
            break
    return threshold

def binarize_with_cdf(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = calc_hist(gray)
    cdf = calculate_cdf(hist)
    cdf_norm = cdf / cdf[-1]

    threshold = get_threshold(cdf_norm, 0.08)
    return black_white_conversion(gray, threshold)
     