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

# Convert to YCrCb format, perform equalization on luminance (Y), return equalized BGR image
def histEqualize(image):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))

    ycrcbImage = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcbImage)

    equalizedY = clahe.apply(y)
    equalizedYcrcbImage = cv.merge([equalizedY, cr, cb])
    equalizedImage = cv.cvtColor(equalizedYcrcbImage, cv.COLOR_YCrCb2BGR)

    return equalizedImage

# Read import image, perform histogram equalization, and convert array to HSV
# Also return size of image
def importImage(path):
    inputImage = cv.imread(path)
    if inputImage is None:
        print(f"Error: Could not load image from {path}")
        return None
    
    width = len(inputImage[0])
    height = len(inputImage)

    equalizedImage = histEqualize(inputImage)
    
    hsvImage = cv.cvtColor(equalizedImage, cv.COLOR_BGR2HSV)
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

        saturation_map[i] = sim.output.get('saturationAdj', 0)
        value_map[i] = sim.output.get('valueAdj', 0)


    return saturation_map, value_map
    

# Antecedent/Consequent ranges
hueRange = np.arange(0,360, 1) # hue range: 0-360
svRange = np.arange(0, 256, 1) # sat/value range: 0-255
deltaHue = np.arange(0.9, 1.1, 0.01)
deltaSaturation = np.arange(-50,50,1)
deltaValue = np.arange(-20,20,1)

# Fuzzy Variables
hueInitial = Antecedent(hueRange, 'hueInitial')
saturationInitial = Antecedent(svRange, 'saturationInitial')
valueInitial = Antecedent(svRange, 'valueInitial')

hueFinal = Consequent(hueRange, 'hueFinal')
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

saturationInitial['dull'] = trapmf(svRange, [0,0,28,46])
saturationInitial['m_dull'] = trapmf(svRange, [28,46,64,82])
saturationInitial['moderate'] = trapmf(svRange, [64,82,138,156])
saturationInitial['m_vivid'] = trapmf(svRange, [138,156,173,191])
saturationInitial['vivid'] = trapmf(svRange, [173,191,255,255])

# 0 = no darkness
# 255 = black
valueInitial['smooth'] = trapmf(svRange, [0,0,55,70])
valueInitial['m_smooth'] = trapmf(svRange, [55,70,95,110])
valueInitial['medium'] = trapmf(svRange, [95,110,140,155])
valueInitial['m_sharp'] = trapmf(svRange, [140,155,180,195])
valueInitial['sharp'] = trapmf(svRange, [180,195,255,255])



# Membership Functions (Output)
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

ruleset = [
    rule1,  rule2,  rule3,  rule4,  rule5, 
    rule6,  rule7,  rule8,  rule9,  rule10,
    rule11, rule12, rule13, rule14, rule15,
    rule16, rule17, rule18, rule19, rule20,
    rule21, rule22, rule23, rule24, rule25
    ]

# Import input image
path = "inputs/coyote.png"
file_name = path.split('/')[1].split('.')[0]
image, width, height = importImage(path)

# # Inference System
controller = ControlSystem(ruleset)
saturation_map, value_map = inferenceSystem(controller)

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

cv.imwrite(f'outputs/{file_name}_enhanced.png', output_image)