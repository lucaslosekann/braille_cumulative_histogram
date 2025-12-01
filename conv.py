import numpy as np

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
