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
import tensorflow as tf
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

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

def rotate_original(to_rotate, centers, image, save):
    #to_rotate is binary braille image with every dot shown just to find orientation

    angle = detect_rotation(centers, to_rotate)
    # print("Raw angle:", angle)

    rotated_original = rotate(image, angle, (255, 255, 255))
    rotated_bin = rotate(to_rotate, angle, (0,0,0))
    contours, _ = cv2.findContours(rotated_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))
        cv2.circle(rotated_bin, (cx, cy), 0, (128))



    if save: cv2.imwrite("imagens_slide/9 - Rotated bin.png", rotated_bin)



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
        
def process_image(image_path, image_frame=None, save=False):
    if image_frame is not None:
        original = image_frame
    else:
        original = cv2.imread(image_path)
    if save: cv2.imwrite("imagens_slide/1 - Original.png", original)
    original = cv2.resize(original, (40, 40), interpolation=cv2.INTER_AREA)
    if save: cv2.imwrite("imagens_slide/2 - Resized.png", original)
    kernel = np.ones((3,3), dtype=np.uint8)

    gray_mask = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if save: cv2.imwrite("imagens_slide/3 - Gray.png", gray_mask)
    gray_mask = cv2.equalizeHist(gray_mask)
    if save: cv2.imwrite("imagens_slide/4 - Equalized.png", gray_mask)
    gray_mask = cv2.GaussianBlur(gray_mask, (5, 5), 0)
    if save: cv2.imwrite("imagens_slide/5 - Blurred.png", gray_mask)
    to_rotate = black_white_conversion(gray_mask, 95)
    # to_rotate = binarize_with_cdf(cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR))
    if save: cv2.imwrite("imagens_slide/6 - Black and White.png", to_rotate)
    open = opening(to_rotate, kernel)
    to_rotate = closing(open, kernel)

    if save: cv2.imwrite("imagens_slide/7 - Morphological.png", to_rotate)

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

    if save: cv2.imwrite("imagens_slide/8 - Centers.png", to_rotate)
    

    

    
    (rotated_original, centers) = rotate_original(to_rotate, centers, original, save=save)
    if save: cv2.imwrite("imagens_slide/10 - Rotated Original.png", rotated_original)
    contrast = cv2.convertScaleAbs(rotated_original, alpha=3, beta=50)
    if save: cv2.imwrite("imagens_slide/11 - Contrast.png", contrast)
    contrast = cv2.equalizeHist(cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY))
    if save: cv2.imwrite("imagens_slide/12 - Equalized Contrast.png", contrast)
    contrast = cv2.GaussianBlur(contrast, (5,5), 0)
    if save: cv2.imwrite("imagens_slide/13 - Blurred Contrast.png", contrast)
    binary = binarize_with_cdf(cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR))
    if save: cv2.imwrite("imagens_slide/14 - Binarized.png", binary)
    dots = detect_braille_dots(binary)
    for bounds in dots:
        x = bounds["w"] // 2 + bounds["x_min"]
        y = bounds["h"] // 2 + bounds["y_min"]
        cv2.rectangle(binary, (x, y), (x, y), (0, 255, 0), 1)
    if save: cv2.imwrite("imagens_slide/15 - Dots Detected.png", binary)

    braille_vector = braille_vector_from_dots(dots, centers)
    char = braille_to_char(braille_vector)

    return (char, braille_vector, dots, binary, contrast, original, rotated_original, to_rotate)

def process_all(method):
    dataset_path = "dataset" if method == "2" else "test_dataset"
    result_dataframe = pd.DataFrame(columns=["filename", "detected_char", "correct_char", "result", "mod"])
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            mod = filename.split(".")[1][-3:]  # 3 last chars before extension
            # if mod == "rot":
            #     continue  
            # if mod == "whs":
            #     continue
            image_path = os.path.join(dataset_path, filename)
            if method == "2":
                (char, braille_vector, dots, binary, contrast, original, rotated_original, to_rotate) = process_image(image_path)
            else:
                char = cnn_predict(image_path)

            correct_char = filename[0]  # first character of filename is the correct one
            result = "correct" if char == correct_char else "wrong"
            # if result == "wrong":
            #     print(f"Error in file {filename}: detected '{char}' but expected '{correct_char}'")
            #     break
            result_dataframe = result_dataframe._append({"filename": filename, "detected_char": char, "correct_char": correct_char, "result": result, "mod": mod}, ignore_index=True)


    print(f"Accuracy: {len(result_dataframe[result_dataframe['result'] == 'correct'])/len(result_dataframe) * 100:.2f}%")
    print(f"Most errors by modification: ")
    results = result_dataframe[result_dataframe['result'] == 'wrong'].groupby('mod').size().sort_values(ascending=False)

    print("Mod:")
    for mod, count in results.items():
        print(f"  {mod}: {count} errors ({(count/len(result_dataframe[result_dataframe['mod'] == mod]) * 100):.2f}%)")

def process_single(image_path, image_frame=None, save=False):
    (char, braille_vector, dots, binary, contrast, original, rotated_original, to_rotate) = process_image(image_path, image_frame, save)

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

    if char is not None:
        mp3_fp = BytesIO()
        tts = gTTS(char, lang='pt', tld='com.br')
        tts.write_to_fp(mp3_fp)
        sound = AudioSegment.from_file(BytesIO(mp3_fp.getvalue()), format="mp3")
        play(sound)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
def cnn_predict(image_path, image_frame=None):
    global model

    if image_frame is not None:
        img = image_frame
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    print("Original shape:", img.shape)

    img = cv2.resize(img, (28, 28))
    img = np.expand_dims(img, axis=-1)  #  (28,28) -> (28,28,1)
    img = np.expand_dims(img, axis=0)  # (28,28,1) -> (1,28,28,1)

    predictions = model.predict(img)
    class_id = np.argmax(predictions, axis=1)[0]
    return chr(ord('a') + class_id)


def cnn_process_single(image_path, image_frame=None):
    if image_frame is not None:
        img = image_frame
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    cv2.imshow("Input", cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST))

    class_id = ord(cnn_predict("", img)) - ord('a')

    print("Predicted class:", class_id, " | Letter:", chr(ord('a') + class_id))

    mp3_fp = BytesIO()
    tts = gTTS(chr(ord('a') + class_id), lang='pt', tld='com.br')
    tts.write_to_fp(mp3_fp)
    sound = AudioSegment.from_file(BytesIO(mp3_fp.getvalue()), format="mp3")
    play(sound)



    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_camera(method):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (40, 40 ), interpolation=cv2.INTER_AREA)
        cv2.imshow("Frame", frame)
        #wait for key press to process
        key = cv2.waitKey(1)
        if key == ord('p'):
            if method == "2":
                process_single(None, frame)
            else:
                cnn_process_single(None, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            break
        elif key == 27:  # ESC key to exit
            break
    cap.release()
    cv2.destroyAllWindows()




def menu():
    # Passar o path da imagem
    #    - CNN
    #    - Histograma
    # Tirar foto com a camera
    #    - CNN
    #    - Histograma
    # Rodar dataset de testes
    #    - CNN
    #    - Histograma
    

    while True:
        #Clear console
        print("\033c", end="")

        print("Menu:")
        print("1. Processar imagem (path)")
        print("2. Processar imagem (camera)")
        print("3. Rodar dataset de testes")
        print("4. Sair")
        choice = input("Escolha uma opcao: ")
        if choice == "1":
            method = get_method_choice()
            if method == "3":
                continue
            image_path = input("Digite o path da imagem: ")

            if not os.path.isfile(image_path):
                print("Arquivo nao encontrado. Tente novamente.")
                input("Pressione Enter para continuar...")
                continue

            if method == "2":
                process_single(image_path, save=True)
            else:
                cnn_process_single(image_path)
        elif choice == "2":
            method = get_method_choice()
            if method == "3":
                continue

            process_camera(method)
        elif choice == "3":
            method = get_method_choice()
            if method == "3":
                continue

            print("Rodando dataset de testes...")
            process_all(method)
            input("Pressione Enter para continuar...")
          
        elif choice == "4":
            print("Saindo...")
            break
        else:
            print("Opcao invalida. Tente novamente.")

def get_method_choice():
    while True:
        print("\033c", end="")
        print("Escolha o metodo de reconhecimento:")
        print("1. CNN")
        print("2. Histograma")
        print("3. Voltar")
        method = input("Escolha uma opcao: ")
        if method in ["1", "2", "3"]:
            return method
        else:
            print("Opcao invalida. Tente novamente.")


if __name__ == "__main__":
    global model
    model = tf.keras.models.load_model("./braille_cnn.keras")
    menu()
    # main()