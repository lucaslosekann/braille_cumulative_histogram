import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from braille_map import braille_to_char
import os
import pandas as pd

def add_padding(img, padding_height, padding_width):
    h, w = img.shape

    padded_img = np.zeros((h + padding_height * 2, w + padding_width * 2))
    padded_img[padding_height : h + padding_height, padding_width : w + padding_width] = img

    return padded_img

def erosion_conv(img, kernel, floatOut=False):
    # Get dimensions of the kernel
    k_height, k_width = kernel.shape

    # Get dimensions of the image
    img_height, img_width = img.shape

    # Calculate padding required
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Create a padded version of the image to handle edges
    img = add_padding(img, pad_height, pad_width)

    # Initialize an output image with zeros
    output = np.zeros((img_height, img_width), dtype=float)

    # Perform convolution

    # print(f"Height: {img_height}")
    # print(f"Width: {img_width}")

    # Iterate only on original image
    for i in range(pad_height, img_height + pad_height):
        for j in range(pad_width, img_width + pad_width):
            accumulation = 0
            isContained = True
            for u in range(k_height):
                for v in range(k_width):
                    kernel_pixel = kernel[u,v]
                    img_pixel = img[i - pad_height + u, j - pad_width + v]
                    if kernel_pixel == 1 and img_pixel != 255:
                        isContained = False
                        break
                if not isContained:
                    break
            output[i-pad_height,j-pad_width] = img[i-pad_height,j-pad_width] if isContained else 0
    if(floatOut):
      return output
    else:
      return np.array(output, dtype=np.uint8)

def dilatation_conv(img, kernel, floatOut=False):
    # Get dimensions of the kernel
    k_height, k_width = kernel.shape

    # Get dimensions of the image
    img_height, img_width = img.shape

    # Calculate padding required
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Create a padded version of the image to handle edges
    img = add_padding(img, pad_height, pad_width)

    # Initialize an output image with zeros
    output = np.zeros((img_height, img_width), dtype=float)

    # Perform convolution

    # print(f"Height: {img_height}")
    # print(f"Width: {img_width}")

    # Iterate only on original image
    for i in range(pad_height, img_height + pad_height):
        for j in range(pad_width, img_width + pad_width):
            if(img[i, j] != 255):
              continue

            for u in range(-k_height // 2, k_height // 2):
                for v in range(-k_width // 2, k_width // 2):
                    kernel_pixel = kernel[u + k_height // 2, v + k_width // 2]

                    out_i = i + u - k_height // 2 + 1
                    out_j = j + v - k_width // 2 + 1

                    if out_i < img_height and out_j < img_width and (kernel_pixel == 1 or img[out_i, out_j] == 255):
                      output[out_i, out_j] = 255
    if(floatOut):
      return output
    else:
      return np.array(output, dtype=np.uint8)
    

def opening(img, kernel):
  return dilatation_conv(erosion_conv(img, kernel), kernel)


def closing(img, kernel):
  return erosion_conv(dilatation_conv(img, kernel), kernel)


def black_white_conversion(image, threshold):
    image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < threshold:
                image[i, j] = 255
            else:
                image[i, j] = 0
    return image

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

    threshold = get_threshold(cdf_norm, 0.1)
    return black_white_conversion(gray, threshold)
     
def detect_dot_bounds(binary, y, x):
    H, W = binary.shape

    x_left = x
    while x_left > 0 and binary[y, x_left] == 255:
        x_left -= 1
    x_left += 1 

    x_right = x
    while x_right < W and binary[y, x_right] == 255:
        x_right += 1
    x_right -= 1

    y_up = y
    while y_up > 0 and binary[y_up, x] == 255:
        y_up -= 1
    y_up += 1

    y_down = y
    while y_down < H and binary[y_down, x] == 255:
        y_down += 1
    y_down -= 1

    return {
        "x_min": x_left,
        "x_max": x_right,
        "y_min": y_up,
        "y_max": y_down,
        "w": x_right - x_left + 1,
        "h": y_down - y_up + 1
    }

def braille_vector_from_dots(dots, centers_global):
    centers = []
    for d in dots:
        cx = d["x_min"] + d["w"]//2
        cy = d["y_min"] + d["h"]//2
        centers.append((cx, cy))
    

    if len(centers_global) == 0:
        return [0,0,0,0,0,0]
    
    x_min = min(cx for cx, _ in centers_global)
    x_max = max(cx for cx, _ in centers_global)
    y_min = min(cy for _, cy in centers_global)
    y_max = max(cy for _, cy in centers_global)

    x_mid = (x_min + x_max) / 2

    y_1 = y_min + (y_max - y_min) / 3
    y_2 = y_min + 2 * (y_max - y_min) / 3

    braille = [0,0,0,0,0,0]  


    for cx, cy in centers:
        if cx < x_mid:
            if cy < y_1:
                braille[0] = 1  
            elif cy < y_2:
                braille[1] = 1  
            else:
                braille[2] = 1  

        
        else:
            if cy < y_1:
                braille[3] = 1  
            elif cy < y_2:
                braille[4] = 1  
            else:
                braille[5] = 1  

    return braille

def rotate(img, angle, borderValue):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=borderValue)
    
def find_braille_orientation(img):
    """
    Detect rotation of Braille cell using pairwise dot-angle clustering.
    Works even when dots are circles with no linear structure.
    Expects img to be binary: dots = 255, background = 0.
    Returns angle (degrees) to apply to correct orientation.
    """

    # detect contours for dot centers
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            x,y,w,h = cv2.boundingRect(c)
            cx = x + w/2
            cy = y + h/2
        else:
            cx = M["m10"]/M["m00"]
            cy = M["m01"]/M["m00"]
        centers.append((cx,cy))

    if len(centers) < 2:
        return 0  # no data

    # compute all pairwise angles
    angles = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            x1, y1 = centers[i]
            x2, y2 = centers[j]
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            # normalize to [-90, 90]
            if angle < -90: angle += 180
            if angle > 90:  angle -= 180
            angles.append(angle)

    if len(angles) == 0:
        return 0

    # cluster around the dominant orientation
    hist, bins = np.histogram(angles, bins=180, range=(-90,90))
    dominant_angle = bins[np.argmax(hist)]

    # rotate image by negative of observed angle
    return -dominant_angle

def detect_rotation(centers):
    print(centers)
    #centers is list of 6 (x,y) tuples and we want to find rotation angle of braille cell
    angles = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            x1, y1 = centers[i]
            x2, y2 = centers[j]
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            # normalize to [-90, 90]
            if angle < -90: angle += 180
            if angle > 90:  angle -= 180
            angles.append(angle)
    if len(angles) == 0:
        return 0
    hist, bins = np.histogram(angles, bins=180, range=(-90,90))
    angle = bins[np.argmax(hist)]



    return angle

def rotate_original(to_rotate, centers, image):
    #to_rotate is binary braille image with every dot shown just to find orientation

    angle = detect_rotation(centers)
    print("Raw angle:", angle)

    rotated_original = rotate(image, angle, (255, 255, 255))
    rotated_bin = rotate(to_rotate, angle, (0,0,0))
    contours, _ = cv2.findContours(to_rotate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))
        cv2.circle(rotated_bin, (cx, cy), 0, (128))



    return (rotated_original, centers)

def detect_braille_dots(binary):
    # soma de pixels ativos por linha
    hist_h = np.sum(binary == 255, axis=1)

    # soma de pixels ativos por coluna
    hist_v = np.sum(binary == 255, axis=0)

    # encontrar picos nos histogramas
    peaks_h = find_peaks(np.concatenate(([0], hist_h, [0])), height=2, distance=5)
    peaks_v = find_peaks(np.concatenate(([0], hist_v, [0])), height=2, distance=5)
    dots = []
    for h in peaks_h[0]:
        for v in peaks_v[0]:
            # corrige padding
            if h <= 0: h = 1
            if v <= 0: v = 1
            if h >= binary.shape[0]: h = binary.shape[0]-1
            if v >= binary.shape[1]: v = binary.shape[1]-1

            if binary[h, v] == 255:  # é um ponto
                bounds = detect_dot_bounds(binary, h, v)
                dots.append(bounds)

    return dots
        


def process_image(image_path):
    kernel = np.ones((3,3), dtype=np.uint8)

    original = cv2.imread(image_path)

    gray_mask = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_mask = cv2.equalizeHist(gray_mask)
    gray_mask = cv2.GaussianBlur(gray_mask, (5, 5), 0)
    to_rotate = black_white_conversion(gray_mask, 100)
    open = opening(to_rotate, kernel)
    to_rotate = closing(open, kernel)
    cv2.imshow("To rotate", to_rotate)
    cv2.imshow("Gray mask", gray_mask)

    contours, _ = cv2.findContours(to_rotate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))
        cv2.circle(to_rotate, (cx, cy), 0, (128))
    

    

    
    # (original, centers) = rotate_original(to_rotate, centers, original)
    contrast = cv2.convertScaleAbs(original, alpha=3, beta=50)
    contrast = cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)
    binary = binarize_with_cdf(contrast)
    dots = detect_braille_dots(binary)
    # print(dots, original)
    braille_vector = braille_vector_from_dots(dots, centers)
    char = braille_to_char(braille_vector)

    return (char, braille_vector, dots, binary, contrast, original, original, to_rotate)

def process_all():
    dataset_path = "dataset"
    result_dataframe = pd.DataFrame(columns=["filename", "detected_char", "correct_char", "result", "mod"])
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            mod = filename.split(".")[1][-3:]  # 3 last chars before extension
            if mod == "rot":
                continue  
            if mod == "whs":
                continue
            image_path = os.path.join(dataset_path, filename)
            (char, braille_vector, dots, binary, contrast, original, rotated_original, to_rotate) = process_image(image_path)
            correct_char = filename[0]  # first character of filename is the correct one
            result = "correct" if char == correct_char else "wrong"
            if result == "wrong":
                print(f"Error in file {filename}: detected '{char}' but expected '{correct_char}'")
                # break
            result_dataframe = result_dataframe._append({"filename": filename, "detected_char": char, "correct_char": correct_char, "result": result, "mod": mod}, ignore_index=True)


    print(f"Accuracy: {len(result_dataframe[result_dataframe['result'] == 'correct'])/len(result_dataframe) * 100:.2f}%")
    print(f"Most errors by modification: ")
    print(result_dataframe[result_dataframe['result'] == 'wrong'].groupby('mod').size().sort_values(ascending=False))

def main():
    (char, braille_vector, dots, binary, contrast, original, rotated_original, to_rotate) = process_image("dataset/n1.JPG4dim.jpg")

    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for bounds in dots:
        x = bounds["w"] // 2 + bounds["x_min"]
        y = bounds["h"] // 2 + bounds["y_min"]
        cv2.rectangle(binary, (x, y), (x, y), (0, 255, 0), 1)

    cv2.imshow("Bin", binary)
    cv2.imshow("Contrast", contrast)
    cv2.imshow("Original", original)
    print("Braille Vector", braille_vector)
    print("Char", char)


    process_all()
    



    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()