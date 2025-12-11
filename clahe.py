import cv2 as cv

from skimage.metrics import normalized_root_mse as nrmse
from fuzzy import getImageNames, histEqualize, computeEntropy, computeTenengrad
from histogram_equalization import importImage


def applyCLAHE(name):
    path = f"images_data/inputs_resized/{name}.png"
    input_image = importImage(path)
    output_image = histEqualize(input_image)

    return input_image, output_image


# Main code block
if __name__ == '__main__':
    image_names = getImageNames('images_data/inputs_resized/')
    for name in image_names:
        input_image, output_image = applyCLAHE(name)

        # Save output image
        name = name.split('_')[0]
        cv.imwrite(f'images_data/outputs/clahe/{name}_enhanced.png', output_image)
        
        # Metrics analysis
        image_nrmse = nrmse(input_image, output_image)
        image_entropy = computeEntropy(output_image) - computeEntropy(input_image)
        image_tenengrad = computeTenengrad(output_image) - computeTenengrad(input_image)

        print(f"{name}:")
        print("NRMSE:", image_nrmse)
        print("Shannon Entropy:", image_entropy)
        print("Tenengrad Score:", image_tenengrad)
        print("\n")