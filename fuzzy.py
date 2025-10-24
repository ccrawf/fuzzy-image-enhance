import numpy as np
import cv2 as cv

from skfuzzy.control import Rule
from skfuzzy.control import ControlSystem
from skfuzzy.control import ControlSystemSimulation
from skfuzzy.control import Antecedent
from skfuzzy.control import Consequent
from skfuzzy.defuzzify import defuzz
from skfuzzy.membership import trapmf
from skfuzzy.image import nmse

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
    

# Variables
hueRange = np.arange(0,180, 1) # hue range: 0-179
svRange = np.arange(0, 256, 1) # sat/value range: 0-255

# Fuzzy Variables
hueInitial = Antecedent(hueRange, 'hue')
saturationInitial = Antecedent(svRange, 'saturation')
valueInitial = Antecedent(svRange, 'value')

hueFinal = Consequent(hueRange, 'hue')
saturationFinal = Consequent(svRange, 'saturation')
valueFinal = Consequent(svRange, 'value')

# Membership Functions
hueInitial('low') = trapmf(hueRange, [0,0,45,75])
hueInitial('medium') = trapmf(hueRange, [45,75,105,135])
hueInitial('high') = trapmf(hueRange, [105,135,179,179])

saturationInitial('dull') = trapmf(svRange, [0,0,70,90])
saturationInitial('moderate') = trapmf(svRange, [70,90,165,185])
saturationInitial('vivid') = trapmf(svRange, [165,185,255,255])

valueInitial('smooth') = trapmf(svRange, [0,0,70,90])
valueInitial('medium') = trapmf(svRange, [70,90,165,185])
valueInitial('sharp') = trapmf(svRange, [165,185,255,255])

image, width, height = importImage("./random-colors-2.png")

