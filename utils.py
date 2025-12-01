def black_white_conversion(image, threshold):
    image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < threshold:
                image[i, j] = 255
            else:
                image[i, j] = 0
    return image
