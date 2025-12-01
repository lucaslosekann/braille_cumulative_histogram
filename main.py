import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from braille_map import braille_to_char
import os
import pandas as pd
from conv import opening, closing
from cdf import binarize_with_cdf
from utils import black_white_conversion


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

def detect_rotation(centers, to_rotate):                        
    # print(centers)
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
    # print("Detected angle:", angle)

    #display histogram
    # plt.hist(angles, bins=180, range=(-90,90))
    # plt.title("Pairwise Dot Angle Histogram")
    # plt.xlabel("Angle (degrees)")
    # plt.ylabel("Frequency")
    # plt.axvline(x=angle, color='r', linestyle='--')
    # plt.show()


    #Now the braile may be horizontally aligned aligned, in that case we need to rotate by 90 degrees or -90 degrees
    rotated_bin = rotate(to_rotate, angle, (0,0,0))
    contours, _ = cv2.findContours(rotated_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers_rotated = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers_rotated.append((cx, cy))
        cv2.circle(rotated_bin, (cx, cy), 0, (128))
    # print("Centers rotated:", centers_rotated)
    x_coords = [c[0] for c in centers_rotated]
    x_range = max(x_coords) - min(x_coords)
    y_coords = [c[1] for c in centers_rotated]
    y_range = max(y_coords) - min(y_coords)
    # print("X range:", x_range, "Y range:", y_range)
    if y_range < x_range:
        if angle > 0:
            angle -= 90
        else:
            angle += 90



    return angle

def rotate_original(to_rotate, centers, image):
    #to_rotate is binary braille image with every dot shown just to find orientation

    angle = detect_rotation(centers, to_rotate)
    # print("Raw angle:", angle)

    rotated_original = rotate(image, angle, (255, 255, 255))
    rotated_bin = rotate(to_rotate, angle, (0,0,0))
    contours, _ = cv2.findContours(to_rotate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))
        cv2.circle(rotated_bin, (cx, cy), 0, (128))

    # cv2.imshow("Rotated bin", rotated_bin)



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
        
def process_image(image_path, image_frame=None):
    if image_frame is not None:
        original = image_frame
    else:
        original = cv2.imread(image_path)
    original = cv2.resize(original, (40, 40), interpolation=cv2.INTER_AREA)

    kernel = np.ones((3,3), dtype=np.uint8)

    gray_mask = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_mask = cv2.equalizeHist(gray_mask)
    gray_mask = cv2.GaussianBlur(gray_mask, (5, 5), 0)
    to_rotate = black_white_conversion(gray_mask, 95)
    open = opening(to_rotate, kernel)
    to_rotate = closing(open, kernel)
    # cv2.imshow("To rotate", to_rotate)
    # cv2.imshow("Gray mask", gray_mask)

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
    # print(centers)
    

    

    
    (rotated_original, centers) = rotate_original(to_rotate, centers, original)
    contrast = cv2.convertScaleAbs(rotated_original, alpha=3, beta=50)
    contrast = cv2.equalizeHist(cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY))
    contrast = cv2.GaussianBlur(contrast, (5,5), 0)
    binary = binarize_with_cdf(cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR))
    dots = detect_braille_dots(binary)
    # print(dots, original)
    braille_vector = braille_vector_from_dots(dots, centers)
    char = braille_to_char(braille_vector)

    return (char, braille_vector, dots, binary, contrast, original, rotated_original, to_rotate)

def process_all():
    dataset_path = "dataset"
    result_dataframe = pd.DataFrame(columns=["filename", "detected_char", "correct_char", "result", "mod"])
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            mod = filename.split(".")[1][-3:]  # 3 last chars before extension
            # if mod == "rot":
            #     continue  
            # if mod == "whs":
            #     continue
            image_path = os.path.join(dataset_path, filename)
            (char, braille_vector, dots, binary, contrast, original, rotated_original, to_rotate) = process_image(image_path)
            correct_char = filename[0]  # first character of filename is the correct one
            result = "correct" if char == correct_char else "wrong"
            # if result == "wrong":
            #     print(f"Error in file {filename}: detected '{char}' but expected '{correct_char}'")
            #     break
            result_dataframe = result_dataframe._append({"filename": filename, "detected_char": char, "correct_char": correct_char, "result": result, "mod": mod}, ignore_index=True)


    print(f"Accuracy: {len(result_dataframe[result_dataframe['result'] == 'correct'])/len(result_dataframe) * 100:.2f}%")
    print(f"Most errors by modification: ")
    print(result_dataframe[result_dataframe['result'] == 'wrong'].groupby('mod').size().sort_values(ascending=False))

def main():
    # (char, braille_vector, dots, binary, contrast, original, rotated_original, to_rotate) = process_image("dataset/j1.JPG8rot.jpg")

    # binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # for bounds in dots:
    #     x = bounds["w"] // 2 + bounds["x_min"]
    #     y = bounds["h"] // 2 + bounds["y_min"]
    #     cv2.rectangle(binary, (x, y), (x, y), (0, 255, 0), 1)

    # cv2.imshow("Bin", binary)
    # cv2.imshow("Contrast", contrast)
    # cv2.imshow("Original", original)
    # print("Braille Vector", braille_vector)
    # print("Char", char)


    process_all()
    




    #get cam video and process each frame
    # while True:
    #     cap = cv2.VideoCapture(0)
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     #scale framte to 50x50 mantaining aspect ratio
    #     frame = cv2.resize(frame, (50, 50 ), interpolation=cv2.INTER_AREA)
    #     (char, braille_vector, dots, binary, contrast, original, rotated_original, to_rotate) = process_image("", frame)
    #     for bounds in dots:
    #         x = bounds["w"] // 2 + bounds["x_min"]
    #         y = bounds["h"] // 2 + bounds["y_min"]
    #         cv2.rectangle(rotated_original, (x, y), (x, y), (0, 255, 0), 1)
    #     print(f"Char: {char}")
    #     cv2.imshow("Rotated Original", rotated_original)
    #     cap.release()
    #     key = cv2.waitKey(1)
    #     if key == 27:  # ESC key to exit
    #         break


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()