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

        saturation_map[i] = sim.output.get('saturationAdj', 0)
        value_map[i] = sim.output.get('valueAdj', 0)


    return saturation_map, value_map
    

# Variables
hueRange = np.arange(0,360, 1) # hue range: 0-360
svRange = np.arange(0, 256, 1) # sat/value range: 0-255
deltaRange = np.arange(-16,16,1) # delta saturation/value

# Fuzzy Variables
hueInitial = Antecedent(hueRange, 'hueInitial')
saturationInitial = Antecedent(svRange, 'saturationInitial')
valueInitial = Antecedent(svRange, 'valueInitial')

hueFinal = Consequent(hueRange, 'hueFinal')
saturationAdj = Consequent(deltaRange, 'saturationAdj')
valueAdj = Consequent(deltaRange, 'valueAdj')

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

saturationAdj['dec_big'] = trapmf(deltaRange, [-16,-16,-8,-4])
saturationAdj['dec_small'] = trapmf(deltaRange, [-8,-4,-1,0])
saturationAdj['no_change'] = trapmf(deltaRange, [-1,0,0,1])
saturationAdj['inc_small'] = trapmf(deltaRange, [0,1,4,8])
saturationAdj['inc_big'] = trapmf(deltaRange, [4,8,50,50])

valueAdj['dec_big'] = trapmf(deltaRange, [-16,-16,-8,-4])
valueAdj['dec_small'] = trapmf(deltaRange, [-8,-4,-1,0])
valueAdj['no_change'] = trapmf(deltaRange, [-1,0,0,1])
valueAdj['inc_small'] = trapmf(deltaRange, [0,1,4,8])
valueAdj['inc_big'] = trapmf(deltaRange, [4,8,50,50])



# Rules
rule1 = Rule(saturationInitial['dull'], saturationAdj['inc_big'])
rule2 = Rule(saturationInitial['m_dull'], saturationAdj['inc_big'])
rule3 = Rule(saturationInitial['moderate'], saturationAdj['inc_big'])
rule4 = Rule(saturationInitial['m_vivid'], saturationAdj['inc_big'])
rule5 = Rule(saturationInitial['vivid'], saturationAdj['no_change'])

rule6 = Rule(valueInitial['smooth'], valueAdj['inc_big'])
rule7 = Rule(valueInitial['m_smooth'], valueAdj['inc_big'])
rule8 = Rule(valueInitial['medium'], valueAdj['inc_big'])
rule9 = Rule(valueInitial['m_sharp'], valueAdj['inc_big'])
rule10 = Rule(valueInitial['sharp'], valueAdj['no_change'])
ruleset = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10]

# Import input image
path = "inputs/man.png"
file_name = path.split('/')[1].split('.')[0]
image, width, height = importImage(path)

# Inference System
controller = ControlSystem(ruleset)
saturation_map, value_map = inferenceSystem(controller)

# saturationAdj.view()
# valueAdj.view()

# plt.show()

print(image[0])

for i, pixel in enumerate(image):
    h, s, v = pixel
    s_delta = saturation_map[int(s)]
    v_delta = value_map[int(v)]

    s_final = int(s + s_delta)
    v_final = int(v + v_delta)

    # print(s, s_final)

    image[i] = (h, s_final, v_final)

# Export output image
image_3d = image.reshape(height, width, 3)
output_image = cv.cvtColor(image_3d, cv.COLOR_HSV2BGR)

print(output_image[0])

cv.imwrite(f'outputs/{file_name}_enhanced.png', output_image)