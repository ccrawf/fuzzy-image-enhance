import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse as nrmse
from fuzzy import getImageNames, computeEntropy, computeTenengrad
from histogram_equalization import importImage

def applyCS(name):
    path = f"images_data/inputs_resized/{name}.png"
    input_image = importImage(path)

    ycrcb_image = cv.cvtColor(input_image, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb_image)

    stretched_y = cv.normalize(y, None, 0, 255, cv.NORM_MINMAX)
    stretched_ycrcb = cv.merge([stretched_y, cr, cb])
    stretched_image = cv.cvtColor(stretched_ycrcb, cv.COLOR_YCrCb2BGR)

    return input_image, stretched_image

# Main code block
if __name__ == '__main__':
    image_names = getImageNames('images_data/inputs_resized/')
    for name in image_names:
        input_image, output_image = applyCS(name)

        # Save output image
        name = name.split('_')[0]
        cv.imwrite(f'images_data/outputs/agcwd/{name}_enhanced.png', output_image)
        
        # Metrics analysis
        image_nrmse = nrmse(input_image, output_image)
        image_entropy = computeEntropy(output_image) - computeEntropy(input_image)
        image_tenengrad = computeTenengrad(output_image)

        print(f"{name}:")
        print("NRMSE:", image_nrmse)
        print("Shannon Entropy:", image_entropy)
        print("Tenengrad Score:", image_tenengrad)
        print("\n")