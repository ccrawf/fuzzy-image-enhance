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
def importImage(path):
    input_image = cv.imread(path)
    if input_image is None:
        print(f"Error: Could not load image from {path}")
        return None
    
    width = len(input_image[0])
    height = len(input_image)

    # Call other function to apply CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    equalized_image = histEqualize(input_image)
    
    hsv_image = cv.cvtColor(equalized_image, cv.COLOR_BGR2HSV)
    reshaped_image = hsv_image.reshape(-1, hsv_image.shape[2])
    return input_image, reshaped_image, width, height

# Compute an output delta value for all possible input values based on rule set
def inferenceSystem(controller):
    sim = ControlSystemSimulation(controller)

    hue_map = np.zeros(360)
    saturation_map = np.zeros(256)
    value_map = np.zeros(256)
    
    for i in range(256):
        sim.reset()
        sim.input['hueInitial'] = i
        sim.input['saturationInitial'] = i
        sim.input['valueInitial'] = i
        sim.compute()

        hue_map[i] = sim.output.get('hueAdj', 0)
        saturation_map[i] = sim.output.get('saturationAdj', 0)
        value_map[i] = sim.output.get('valueAdj', 0)


    return hue_map, saturation_map, value_map

# Full image transformation 
def transformImage(image_name):
    # Import input image
    path = f"images_data/inputs/{image_name}.png"
    input_image, image, width, height = importImage(path)

    # Inference System
    controller = ControlSystem(ruleset)
    hue_map, saturation_map, value_map = inferenceSystem(controller)

    for i, pixel in enumerate(image):
        h, s, v = pixel
        h_delta = hue_map[int(h)]
        s_delta = saturation_map[int(s)]
        v_delta = value_map[int(v)]

        h_final = int(h + h_delta)
        s_final = int(s + s_delta)
        v_final = int(v + v_delta)

        image[i] = (h_final, s_final, v_final)

    # Export output image
    image_3d = image.reshape(height, width, 3)
    output_image = cv.cvtColor(image_3d, cv.COLOR_HSV2BGR)

    cv.imwrite(f'images_data/outputs/{image_name}_enhanced.png', output_image)

    return input_image, output_image

# Display current membership functions (used for testing only)
def printMembershipFunctions():
    hueInitial.view()
    saturationInitial.view()
    valueInitial.view()
    hueAdj.view()
    saturationAdj.view()
    valueAdj.view()

    plt.show()

# Returns list of file names in input image directory
def getImageNames(path):
    image_names = []

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            item = item.split('.')[0]
            image_names.append(item)

    return image_names

# Antecedent/Consequent ranges
hueRange = np.arange(0,360, 1) # hue range: 0-360
svRange = np.arange(0, 256, 1) # sat/value range: 0-255

deltaHue = np.arange(-20, 20, 1)
deltaSaturation = np.arange(-50,50,1)
deltaValue = np.arange(-20,20,1)

# Fuzzy Variables
hueInitial = Antecedent(hueRange, 'hueInitial')
saturationInitial = Antecedent(svRange, 'saturationInitial')
valueInitial = Antecedent(svRange, 'valueInitial')

hueAdj = Consequent(deltaHue, 'hueAdj')
saturationAdj = Consequent(deltaSaturation, 'saturationAdj')
valueAdj = Consequent(deltaValue, 'valueAdj')

# Membership Functions (Input)
hueInitial['red'] = trapmf(hueRange, [0,0,10,15])
hueInitial['brown'] = trapmf(hueRange, [10,15,18,23])
hueInitial['orange'] = trapmf(hueRange, [18,23,35,40])
hueInitial['yellow'] = trapmf(hueRange, [35,40,60,90])
hueInitial['green'] = trapmf(hueRange, [70,90,130,160])
hueInitial['cyan'] = trapmf(hueRange, [140,160,190,210])
hueInitial['blue'] = trapmf(hueRange, [190,210,240,270])
hueInitial['purple'] = trapmf(hueRange, [255,270,290,305])
hueInitial['pink'] = trapmf(hueRange, [295,305,330,345])
hueInitial['red2'] = trapmf(hueRange, [330,345,359,359])

saturationInitial['dull'] = trapmf(svRange, [0,0,28,46]) # no color
saturationInitial['m_dull'] = trapmf(svRange, [28,46,64,82])
saturationInitial['moderate'] = trapmf(svRange, [64,82,138,156])
saturationInitial['m_vivid'] = trapmf(svRange, [138,156,173,191])
saturationInitial['vivid'] = trapmf(svRange, [173,191,255,255]) # full color

valueInitial['smooth'] = trapmf(svRange, [0,0,55,70]) # no darkness
valueInitial['m_smooth'] = trapmf(svRange, [55,70,95,110])
valueInitial['medium'] = trapmf(svRange, [95,110,140,155])
valueInitial['m_sharp'] = trapmf(svRange, [140,155,180,195])
valueInitial['sharp'] = trapmf(svRange, [180,195,255,255]) # black

# Membership Functions (Output)
hueAdj['dec_big'] = trapmf(deltaHue, [-20,-20,-12,-10])
hueAdj['dec_small'] = trapmf(deltaHue, [-12,-10,-1,0])
hueAdj['no_change'] = trapmf(deltaHue, [-1,0,0,1])
hueAdj['inc_small'] = trapmf(deltaHue, [0,1,10,12])
hueAdj['inc_big'] = trapmf(deltaHue, [10,12,20,20])

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
rule1 = Rule(saturationInitial['dull'] & valueInitial['smooth'], (saturationAdj['inc_small'], valueAdj['inc_big']))
rule2 = Rule(saturationInitial['dull'] & valueInitial['m_smooth'], (saturationAdj['inc_big'], valueAdj['inc_small']))
rule3 = Rule(saturationInitial['dull'] & valueInitial['medium'], (saturationAdj['inc_big'], valueAdj['inc_small']))
rule4 = Rule(saturationInitial['dull'] & valueInitial['m_sharp'], (saturationAdj['inc_big'], valueAdj['no_change']))
rule5 = Rule(saturationInitial['dull'] & valueInitial['sharp'], (saturationAdj['inc_big'], valueAdj['no_change']))

rule6 = Rule(saturationInitial['m_dull'] & valueInitial['smooth'], (saturationAdj['no_change'], valueAdj['inc_big']))
rule7 = Rule(saturationInitial['m_dull'] & valueInitial['m_smooth'], (saturationAdj['inc_small'], valueAdj['inc_small']))
rule8 = Rule(saturationInitial['m_dull'] & valueInitial['medium'], (saturationAdj['inc_big'], valueAdj['no_change']))
rule9 = Rule(saturationInitial['m_dull'] & valueInitial['m_sharp'], (saturationAdj['inc_big'], valueAdj['no_change']))
rule10 = Rule(saturationInitial['m_dull'] & valueInitial['sharp'], (saturationAdj['inc_big'], valueAdj['dec_small']))

rule11 = Rule(saturationInitial['moderate'] & valueInitial['smooth'], (saturationAdj['no_change'], valueAdj['inc_small']))
rule12 = Rule(saturationInitial['moderate'] & valueInitial['m_smooth'], (saturationAdj['no_change'], valueAdj['no_change']))
rule13 = Rule(saturationInitial['moderate'] & valueInitial['medium'], (saturationAdj['inc_small'], valueAdj['no_change']))
rule14 = Rule(saturationInitial['moderate'] & valueInitial['m_sharp'], (saturationAdj['inc_small'], valueAdj['no_change']))
rule15 = Rule(saturationInitial['moderate'] & valueInitial['sharp'], (saturationAdj['inc_small'], valueAdj['dec_small']))

rule16 = Rule(saturationInitial['m_vivid'] & valueInitial['smooth'], (saturationAdj['no_change'], valueAdj['inc_small']))
rule17 = Rule(saturationInitial['m_vivid'] & valueInitial['m_smooth'], (saturationAdj['no_change'], valueAdj['no_change']))
rule18 = Rule(saturationInitial['m_vivid'] & valueInitial['medium'], (saturationAdj['no_change'], valueAdj['dec_small']))
rule19 = Rule(saturationInitial['m_vivid'] & valueInitial['m_sharp'], (saturationAdj['dec_small'], valueAdj['dec_small']))
rule20 = Rule(saturationInitial['m_vivid'] & valueInitial['sharp'], (saturationAdj['dec_small'], valueAdj['dec_big']))

rule21 = Rule(saturationInitial['vivid'] & valueInitial['smooth'], (saturationAdj['no_change'], valueAdj['inc_small']))
rule22 = Rule(saturationInitial['vivid'] & valueInitial['m_smooth'], (saturationAdj['dec_small'], valueAdj['no_change']))
rule23 = Rule(saturationInitial['vivid'] & valueInitial['medium'], (saturationAdj['dec_small'], valueAdj['dec_small']))
rule24 = Rule(saturationInitial['vivid'] & valueInitial['m_sharp'], (saturationAdj['dec_big'], valueAdj['dec_big']))
rule25 = Rule(saturationInitial['vivid'] & valueInitial['sharp'], (saturationAdj['dec_big'], valueAdj['dec_big']))

# Additional rules (edge cases with hue)
rule26 = Rule(hueInitial['red'] |
 hueInitial['brown'] |
 hueInitial['orange'] |
 hueInitial['yellow'] |
 hueInitial['green'] |
 hueInitial['cyan'] |
 hueInitial['blue'] |
 hueInitial['purple'] |
 hueInitial['pink'] |
 hueInitial['red2'], 
 hueAdj['no_change'])

ruleset = [
    rule1,  rule2,  rule3,  rule4,  rule5, 
    rule6,  rule7,  rule8,  rule9,  rule10,
    rule11, rule12, rule13, rule14, rule15,
    rule16, rule17, rule18, rule19, rule20,
    rule21, rule22, rule23, rule24, rule25,
    rule26
    ]

image_names = getImageNames('images_data/inputs/')
for name in image_names:
    # Fuzzy image transform
    input_image, output_image = transformImage(name)

    # Metrics analysis
    image_nrmse = nrmse(input_image, output_image)
    image_entropy = abs(shannon_entropy(output_image) - shannon_entropy(input_image))

    print(f"{name}:")
    print("NRMSE:", image_nrmse)
    print("Shannon Entropy:", image_entropy)
    print("Tenengrad Score:")
    print("\n")