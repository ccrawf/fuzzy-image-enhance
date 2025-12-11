import numpy as np
import cv2 as cv
import os

from skfuzzy.control import Rule
from skfuzzy.control import ControlSystem
from skfuzzy.control import ControlSystemSimulation
from skfuzzy.control import Antecedent
from skfuzzy.control import Consequent
from skfuzzy.membership import trapmf
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse as nrmse
from skimage.measure import shannon_entropy

# Returns list of file names in input image directory
def getImageNames(path):
    image_names = []

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            item = item.split('.')[0]
            image_names.append(item)

    return image_names

# Resize all images into 500x500
def resize(image, name):
    w, h = 500, 500
    resized_image = cv.resize(image, (w, h))
    cv.imwrite(f"images_data/inputs_resized/{name}_resized.png", resized_image)

    return resized_image

# Convert to YCrCb format, perform equalization on luminance (Y), return equalized BGR image
def histEqualize(image):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))

    ycrcb_image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb_image)

    equalized_y = clahe.apply(y)
    equalized_ycrcb_image = cv.merge([equalized_y, cr, cb])
    equalized_image = cv.cvtColor(equalized_ycrcb_image, cv.COLOR_YCrCb2BGR)

    return equalized_image

# Read import image, perform histogram equalization, and convert array to HSV
# Return size of image
def importImage(path, name):
    input_image = cv.imread(path)
    if input_image is None:
        print(f"Error: Could not load image from {path}")
        return None

    # Call other function to apply CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    input_image = resize(input_image, name)
    equalized_image = histEqualize(input_image)

    width = len(input_image[0])
    height = len(input_image)
    
    hsv_image = cv.cvtColor(equalized_image, cv.COLOR_BGR2HSV)
    reshaped_image = hsv_image.reshape(-1, hsv_image.shape[2])
    return input_image, reshaped_image, width, height

# Compute an output delta value for all possible input values based on rule set
def inferenceSystem(controller):
    sim = ControlSystemSimulation(controller)

    saturation_map = np.zeros(256)
    value_map = np.zeros(256)
    
    for i in range(256):
        sim.reset()
        sim.input['hueInitial'] = i
        sim.input['saturationInitial'] = i
        sim.input['valueInitial'] = i
        sim.compute()

        saturation_map[i] = sim.output.get('saturationAdj', 0)
        value_map[i] = sim.output.get('valueAdj', 0)


    return saturation_map, value_map

# Full image transformation 
def transformImage(image_name):
    # Import input image
    path = f"images_data/inputs/{image_name}.png"
    input_image, image, width, height = importImage(path, image_name)
    
    for i, pixel in enumerate(image):
        h, s, v = pixel
        s_delta = saturation_map[int(s)]
        v_delta = value_map[int(v)]

        s_final = int(s + s_delta)
        v_final = int(v + v_delta)

        image[i] = (h, s_final, v_final)

    # Export output image
    image_3d = image.reshape(height, width, 3)
    output_image = cv.cvtColor(image_3d, cv.COLOR_HSV2BGR)

    cv.imwrite(f'images_data/outputs/fuzzy/{image_name}_enhanced.png', output_image)

    return input_image, output_image

# Display current membership functions (used for testing only)
def printMembershipFunctions():
    hueInitial.view()
    saturationInitial.view()
    valueInitial.view()
    saturationAdj.view()
    valueAdj.view()

    plt.show()

# Metrics
# Returns Shannon Entropy score for the grayscale image
def computeEntropy(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return shannon_entropy(image_gray)

# Returns Tenengrad score (Sobel Gradient based, measures strong/sharp edges)
def computeTenengrad(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sobel_x = cv.Sobel(image_gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(image_gray, cv.CV_64F, 0, 1, ksize=3)

    tenengrad = np.sqrt(sobel_x**2 + sobel_y**2)

    return np.mean(tenengrad)

# Antecedent/Consequent ranges
hueRange = np.arange(0,180, 1) # hue range: 0-180
svRange = np.arange(0, 256, 1) # sat/value range: 0-255

deltaSaturation = np.arange(-50,50,1)
deltaValue = np.arange(-20,20,1)

# Fuzzy Variables
hueInitial = Antecedent(hueRange, 'hueInitial')
saturationInitial = Antecedent(svRange, 'saturationInitial')
valueInitial = Antecedent(svRange, 'valueInitial')

saturationAdj = Consequent(deltaSaturation, 'saturationAdj')
valueAdj = Consequent(deltaValue, 'valueAdj')

# Membership Functions (Input)
hueInitial['red'] = trapmf(hueRange, [0,0,5,8])
hueInitial['brown'] = trapmf(hueRange, [5,8,9,12])
hueInitial['orange'] = trapmf(hueRange, [9,12,18,20])
hueInitial['yellow'] = trapmf(hueRange, [18,20,30,40])
hueInitial['green'] = trapmf(hueRange, [30,40,65,80])
hueInitial['cyan'] = trapmf(hueRange, [70,80,95,105])
hueInitial['blue'] = trapmf(hueRange, [95,105,120,135])
hueInitial['purple'] = trapmf(hueRange, [128,135,145,153])
hueInitial['pink'] = trapmf(hueRange, [148,153,165,172])
hueInitial['red2'] = trapmf(hueRange, [165,172,179,179])

saturationInitial['dull'] = trapmf(svRange, [0,0,28,46]) # no color
saturationInitial['m_dull'] = trapmf(svRange, [28,46,64,82])
saturationInitial['moderate'] = trapmf(svRange, [64,82,138,156])
saturationInitial['m_vivid'] = trapmf(svRange, [138,156,173,191])
saturationInitial['vivid'] = trapmf(svRange, [173,191,255,255]) # full color

valueInitial['dark'] = trapmf(svRange, [0,0,55,70]) # black
valueInitial['m_dark'] = trapmf(svRange, [55,70,95,110])
valueInitial['medium'] = trapmf(svRange, [95,110,140,155])
valueInitial['m_bright'] = trapmf(svRange, [140,155,180,195])
valueInitial['bright'] = trapmf(svRange, [180,195,255,255]) # no darkness

# Membership Functions (Output)
saturationAdj['dec_big'] = trapmf(deltaSaturation, [-50,-50,-30,-25])
saturationAdj['dec_small'] = trapmf(deltaSaturation, [-30,-25,-1,0])
saturationAdj['no_change'] = trapmf(deltaSaturation, [-1,0,0,1])
saturationAdj['inc_small'] = trapmf(deltaSaturation, [0,1,25,30])
saturationAdj['inc_big'] = trapmf(deltaSaturation, [25,30,50,50])

valueAdj['dec_big'] = trapmf(deltaValue, [-20,-20,-16,-12])
valueAdj['dec_small'] = trapmf(deltaValue, [-16,-12,-1,0])
valueAdj['no_change'] = trapmf(deltaValue, [-1,0,0,1])
valueAdj['inc_small'] = trapmf(deltaValue, [0,1,12,16])
valueAdj['inc_big'] = trapmf(deltaValue, [12,16,20,20])

# Rules
# All possible combinations of S and V
rule1 = Rule(saturationInitial['dull'] & valueInitial['dark'], (saturationAdj['inc_small'], valueAdj['inc_big']))
rule2 = Rule(saturationInitial['dull'] & valueInitial['m_dark'], (saturationAdj['inc_big'], valueAdj['inc_small']))
rule3 = Rule(saturationInitial['dull'] & valueInitial['medium'], (saturationAdj['inc_big'], valueAdj['inc_small']))
rule4 = Rule(saturationInitial['dull'] & valueInitial['m_bright'], (saturationAdj['inc_big'], valueAdj['no_change']))
rule5 = Rule(saturationInitial['dull'] & valueInitial['bright'], (saturationAdj['inc_big'], valueAdj['no_change']))

rule6 = Rule(saturationInitial['m_dull'] & valueInitial['dark'], (saturationAdj['no_change'], valueAdj['inc_big']))
rule7 = Rule(saturationInitial['m_dull'] & valueInitial['m_dark'], (saturationAdj['inc_small'], valueAdj['inc_small']))
rule8 = Rule(saturationInitial['m_dull'] & valueInitial['medium'], (saturationAdj['inc_big'], valueAdj['no_change']))
rule9 = Rule(saturationInitial['m_dull'] & valueInitial['m_bright'], (saturationAdj['inc_big'], valueAdj['no_change']))
rule10 = Rule(saturationInitial['m_dull'] & valueInitial['bright'], (saturationAdj['inc_big'], valueAdj['dec_small']))

rule11 = Rule(saturationInitial['moderate'] & valueInitial['dark'], (saturationAdj['no_change'], valueAdj['inc_small']))
rule12 = Rule(saturationInitial['moderate'] & valueInitial['m_dark'], (saturationAdj['no_change'], valueAdj['no_change']))
rule13 = Rule(saturationInitial['moderate'] & valueInitial['medium'], (saturationAdj['inc_small'], valueAdj['no_change']))
rule14 = Rule(saturationInitial['moderate'] & valueInitial['m_bright'], (saturationAdj['inc_small'], valueAdj['no_change']))
rule15 = Rule(saturationInitial['moderate'] & valueInitial['bright'], (saturationAdj['inc_small'], valueAdj['dec_small']))

rule16 = Rule(saturationInitial['m_vivid'] & valueInitial['dark'], (saturationAdj['no_change'], valueAdj['inc_small']))
rule17 = Rule(saturationInitial['m_vivid'] & valueInitial['m_dark'], (saturationAdj['no_change'], valueAdj['no_change']))
rule18 = Rule(saturationInitial['m_vivid'] & valueInitial['medium'], (saturationAdj['no_change'], valueAdj['dec_small']))
rule19 = Rule(saturationInitial['m_vivid'] & valueInitial['m_bright'], (saturationAdj['dec_small'], valueAdj['dec_small']))
rule20 = Rule(saturationInitial['m_vivid'] & valueInitial['bright'], (saturationAdj['dec_small'], valueAdj['dec_big']))

rule21 = Rule(saturationInitial['vivid'] & valueInitial['dark'], (saturationAdj['no_change'], valueAdj['inc_small']))
rule22 = Rule(saturationInitial['vivid'] & valueInitial['m_dark'], (saturationAdj['dec_small'], valueAdj['no_change']))
rule23 = Rule(saturationInitial['vivid'] & valueInitial['medium'], (saturationAdj['dec_small'], valueAdj['dec_small']))
rule24 = Rule(saturationInitial['vivid'] & valueInitial['m_bright'], (saturationAdj['dec_big'], valueAdj['dec_big']))
rule25 = Rule(saturationInitial['vivid'] & valueInitial['bright'], (saturationAdj['dec_big'], valueAdj['dec_big']))

# Additional rules (edge cases with hue)
rule26 = Rule((hueInitial['blue'] | hueInitial['purple']), 
              valueAdj['no_change'])

rule27 = Rule((hueInitial['brown'] | hueInitial['orange'] | hueInitial['yellow'] | hueInitial['pink']) & 
              (valueInitial['bright'] | valueInitial['m_bright']), 
              valueAdj['dec_big'])

rule28 = Rule(hueInitial['green'] | hueInitial['cyan'], 
              saturationAdj['inc_small'])

ruleset = [
    rule1,  rule2,  rule3,  rule4,  rule5, 
    rule6,  rule7,  rule8,  rule9,  rule10,
    rule11, rule12, rule13, rule14, rule15,
    rule16, rule17, rule18, rule19, rule20,
    rule21, rule22, rule23, rule24, rule25,
    rule26, rule27, rule28
    ]

# Main code block
if __name__ == '__main__':
    # Inference System
    controller = ControlSystem(ruleset)
    saturation_map, value_map = inferenceSystem(controller)

    image_names = getImageNames('images_data/inputs/')
    for name in image_names:
        # Fuzzy image transform
        input_image, output_image = transformImage(name)

        # Metrics analysis
        image_nrmse = nrmse(input_image, output_image)
        image_entropy = computeEntropy(output_image) - computeEntropy(input_image)
        image_tenengrad = computeTenengrad(output_image) - computeTenengrad(input_image)

        print(f"{name}:")
        print("NRMSE:", image_nrmse)
        print("Shannon Entropy:", image_entropy)
        print("Tenengrad Score:", image_tenengrad)
        print("\n")