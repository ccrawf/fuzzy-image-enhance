import numpy as np
import cv2 as cv

from skfuzzy.control import Rule
from skfuzzy.control import ControlSystem
from skfuzzy.control import ControlSystemSimulation
from skfuzzy.control import Antecedent
from skfuzzy.control import Consequent
from skfuzzy.defuzzify import defuzz
from skfuzzy.membership import trapmf
import matplotlib.pyplot as plt
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
hueRange = np.arange(0,360, 1) # hue range: 0-360
svRange = np.arange(0, 256, 1) # sat/value range: 0-255

# Fuzzy Variables
hueInitial = Antecedent(hueRange, 'hueInitial')
saturationInitial = Antecedent(svRange, 'saturationInitial')
valueInitial = Antecedent(svRange, 'valueInitial')

hueFinal = Consequent(hueRange, 'hueFinal')
saturationFinal = Consequent(svRange, 'saturationFinal')
valueFinal = Consequent(svRange, 'valueFinal')

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

saturationInitial['dull'] = trapmf(svRange, [0,0,28,46])
saturationInitial['m_dull'] = trapmf(svRange, [28,46,64,82])
saturationInitial['moderate'] = trapmf(svRange, [64,82,138,156])
saturationInitial['m_vivid'] = trapmf(svRange, [138,156,173,191])
saturationInitial['vivid'] = trapmf(svRange, [173,191,255,255])

# 0 = black
# 255 = no darkness
valueInitial['smooth'] = trapmf(svRange, [0,0,55,70])
valueInitial['m_smooth'] = trapmf(svRange, [55,70,95,110])
valueInitial['medium'] = trapmf(svRange, [95,110,140,155])
valueInitial['m_sharp'] = trapmf(svRange, [140,155,180,195])
valueInitial['sharp'] = trapmf(svRange, [180,195,255,255])

# Membership Functions (Output)
# Note: output functions are equivalent to input
hueFinal['red'] = trapmf(hueRange, [0,0,10,15])
hueFinal['brown'] = trapmf(hueRange, [10,15,18,23])
hueFinal['orange'] = trapmf(hueRange, [18,23,35,40])
hueFinal['yellow'] = trapmf(hueRange, [35,40,60,90])
hueFinal['green'] = trapmf(hueRange, [70,90,130,160])
hueFinal['cyan'] = trapmf(hueRange, [140,160,190,210])
hueFinal['blue'] = trapmf(hueRange, [190,210,240,270])
hueFinal['purple'] = trapmf(hueRange, [255,270,290,305])
hueFinal['pink'] = trapmf(hueRange, [295,305,330,345])
hueFinal['red2'] = trapmf(hueRange, [330,345,359,359])

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

hueInitial.view()
saturationInitial.view()
valueInitial.view()

plt.show()

for i, pixel in enumerate(image):
    h, s, v = pixel
    s2 = saturation_map[int(s)]
    v2 = value_map[int(v)]
    image[i] = (h, s2, v2)

# Export output image
image_3d = image.reshape(300, 600, 3)
output_image = cv.cvtColor(image_3d, cv.COLOR_HSV2BGR)
cv.imwrite('./output.png', output_image)