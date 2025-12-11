# AGCWD Algorithm from github.com/qyou
import numpy as np
import cv2 as cv

from skimage.metrics import normalized_root_mse as nrmse
from fuzzy import getImageNames, computeEntropy, computeTenengrad
from histogram_equalization import importImage

def agcwd(name):
    path = f"images_data/inputs_resized/{name}.png"
    input_image = importImage(path)

    # Extract Value channel of image
    hsv_image = cv.cvtColor(input_image, cv.COLOR_BGR2HSV)
    v_image = hsv_image[:, :, 2]

    # PDF (Probability Distribution Function) of value image
    height, width = v_image.shape
    num_pixels = height * width
    hist = cv.calcHist([v_image], [0], None, [256], [0, 256])
    pdf = hist / num_pixels

    # CDF (Cumulative Distribution Function) of image
    max_intensity = np.max(pdf)
    min_intensity = np.min(pdf)
    img_pdf = max_intensity * (((pdf - min_intensity) / (max_intensity - min_intensity)) ** 0.5)
    img_cdf = np.cumsum(img_pdf) / np.sum(img_pdf)

    # Intensity
    l_intensity = np.arange(0,256)
    l_intensity = np.array([255 * (e / 255) ** (1 - img_cdf[e]) for e in l_intensity], dtype=np.uint8)
    enhanced_image = np.copy(input_image)

    for i in range(0, height):
        for j in range(0, width):
            intensity = enhanced_image[i, j]
            enhanced_image[i, j] = l_intensity[intensity]

    return input_image, enhanced_image

# Main code block
if __name__ == '__main__':
    image_names = getImageNames('images_data/inputs_resized/')
    for name in image_names:
        input_image, output_image = agcwd(name)

        # Save output image
        name = name.split('_')[0]
        cv.imwrite(f'images_data/outputs/agcwd/{name}_enhanced.png', output_image)
        
        # Metrics analysis
        image_nrmse = nrmse(input_image, output_image)
        image_entropy = computeEntropy(output_image) - computeEntropy(input_image)
        image_tenengrad = computeTenengrad(output_image) - computeTenengrad(input_image)

        print(f"{name}:")
        print("NRMSE:", image_nrmse)
        print("Shannon Entropy:", image_entropy)
        print("Tenengrad Score:", image_tenengrad)
        print("\n")