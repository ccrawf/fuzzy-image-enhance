import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse as nrmse
from fuzzy import getImageNames, computeEntropy, computeTenengrad

def importImage(path):
    input_image = cv.imread(path)
    if input_image is None:
        print(f"Error: Could not load image from {path}")
        return None
    
    return input_image
    
# Use OpenCV histogram equalization on luminance (grayscale) channel of image
def applyHistogramEqualization(name):
    path = f"images_data/inputs_resized/{name}.png"
    input_image = importImage(path)

    ycrcb_image = cv.cvtColor(input_image, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb_image)

    equalized_y = cv.equalizeHist(y)
    equalized_ycrcb_image = cv.merge([equalized_y, cr, cb])
    equalized_image = cv.cvtColor(equalized_ycrcb_image, cv.COLOR_YCrCb2BGR)

    return input_image, equalized_image

# Main code block
if __name__ == '__main__':
    image_names = getImageNames('images_data/inputs_resized/')
    for name in image_names:
        input_image, output_image = applyHistogramEqualization(name)

        # Save output image
        name = name.split('_')[0]
        cv.imwrite(f'images_data/outputs/histogram_equalization/{name}_enhanced.png', output_image)
        
        # Metrics analysis
        image_nrmse = nrmse(input_image, output_image)
        image_entropy = computeEntropy(output_image) - computeEntropy(input_image)
        image_tenengrad = computeTenengrad(output_image)

        print(f"{name}:")
        print("NRMSE:", image_nrmse)
        print("Shannon Entropy:", image_entropy)
        print("Tenengrad Score:", image_tenengrad)
        print("\n")