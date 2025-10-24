import numpy as np
import cv2 as cv

from skfuzzy.control import Rule
from skfuzzy.control import ControlSystem
from skfuzzy.control import ControlSystemSimulation
from skfuzzy.control import Antecedent
from skfuzzy.control import Consequent
from skfuzzy.defuzzify import defuzz
from skfuzzy.membership import trapmf
# from skfuzzy.image import nmse

# Read import image and convert array to HSV
# Also return size of image
def importImage(path):
    inputImage = cv.imread(path)
    if inputImage is None:
        print(f"Error: Could not load image from {path}")
        return None
    
    width = len(inputImage[0])
    height = len(inputImage)
    
    hsvImage = cv.cvtColor(inputImage, cv.COLOR_BGR2HSV)
    reshapedImage = hsvImage.reshape(-1, hsvImage.shape[2])
    return reshapedImage, width, height

def inferenceSystem(controller):

    sim = ControlSystemSimulation(controller)

    saturation_map = np.zeros(256)
    value_map = np.zeros(256)
    
    for i in range(256):
        sim.reset()
        sim.input['saturationInitial'] = i
        sim.input['valueInitial'] = i
        sim.compute()

        saturation_map[i] = sim.output['saturationFinal']
        value_map[i] = sim.output['valueFinal']


    return saturation_map, value_map
    

# Variables
hueRange = np.arange(0,180, 1) # hue range: 0-179
svRange = np.arange(0, 256, 1) # sat/value range: 0-255

# Fuzzy Variables
hueInitial = Antecedent(hueRange, 'hueInitial')
saturationInitial = Antecedent(svRange, 'saturationInitial')
valueInitial = Antecedent(svRange, 'valueInitial')

hueFinal = Consequent(hueRange, 'hueFinal')
saturationFinal = Consequent(svRange, 'saturationFinal')
valueFinal = Consequent(svRange, 'valueFinal')

# Membership Functions (Input)
hueInitial['low'] = trapmf(hueRange, [0,0,45,75])
hueInitial['medium'] = trapmf(hueRange, [45,75,105,135])
hueInitial['high'] = trapmf(hueRange, [105,135,179,179])

saturationInitial['dull'] = trapmf(svRange, [0,0,70,90])
saturationInitial['moderate'] = trapmf(svRange, [70,90,165,185])
saturationInitial['vivid'] = trapmf(svRange, [165,185,255,255])

valueInitial['smooth'] = trapmf(svRange, [0,0,70,90])
valueInitial['medium'] = trapmf(svRange, [70,90,165,185])
valueInitial['sharp'] = trapmf(svRange, [165,185,255,255])

# Membership Functions (Output)
# Note: output functions are equivalent to input
hueFinal['low'] = trapmf(hueRange, [0,0,45,75])
hueFinal['medium'] = trapmf(hueRange, [45,75,105,135])
hueFinal['high'] = trapmf(hueRange, [105,135,179,179])

saturationFinal['dull'] = trapmf(svRange, [0,0,70,90])
saturationFinal['moderate'] = trapmf(svRange, [70,90,165,185])
saturationFinal['vivid'] = trapmf(svRange, [165,185,255,255])

valueFinal['smooth'] = trapmf(svRange, [0,0,70,90])
valueFinal['medium'] = trapmf(svRange, [70,90,165,185])
valueFinal['sharp'] = trapmf(svRange, [165,185,255,255])

# Rules
rule1 = Rule(saturationInitial['dull'], saturationFinal['moderate'])
rule2 = Rule(saturationInitial['moderate'], saturationFinal['moderate'])
rule3 = Rule(saturationInitial['vivid'], saturationFinal['moderate'])
rule4 = Rule(valueInitial['smooth'], valueFinal['medium'])
rule5 = Rule(valueInitial['medium'], valueFinal['medium'])
rule6 = Rule(valueInitial['sharp'], valueFinal['medium'])
ruleset = [rule1, rule2, rule3, rule4, rule5, rule6]

# Import input image
image, width, height = importImage("./random-colors-2.png")

# Inference System
controller = ControlSystem(ruleset)
saturation_map, value_map = inferenceSystem(controller)

for i, pixel in enumerate(image):
    h, s, v = pixel
    s2 = saturation_map[int(s)]
    v2 = value_map[int(v)]
    image[i] = (h, s2, v2)

# Export output image
image_3d = image.reshape(300, 600, 3)
output_image = cv.cvtColor(image_3d, cv.COLOR_HSV2BGR)
cv.imwrite('./output.png', output_image)