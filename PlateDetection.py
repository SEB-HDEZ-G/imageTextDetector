import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import easyocr

#Gaussian filter function
def gaussianFilter(img, kernelSize, sigma=0):
    """
    The Gaussian filter talkes as parameters:
    - img: Input image (can be grayscale or color, but I'm using grayscale)
    - kernelSize: Size of the Gaussian kernel (must be odd), larger kernel gives a stronger blur
    - sigma: Standard deviation (how much smoothing is applied), gets handled automatically by OpenCV if the value is 0
    """
    
    return cv.GaussianBlur(img, (kernelSize, kernelSize), sigma)

#STEP 1 - image reading
img = cv.imread("./ImageTextDetector/placa_q.jpg")
img2 = cv.imread("./ImageTextDetector/placa_2.jpg")
imgGroup = [img, img2]
#grayscale conversion can't be used for HSV color filtering, but cv.IMREAD_GRAYSCALE would be used otherwise

#STEPS 2 & 3 - filter white from image and apply Gaussian filter
for i, img in enumerate(imgGroup):
    #noise reduction using Gaussian filter, kernel size of 39
    imgGaussian = gaussianFilter(img, 39)
    
    imgHSV = cv.cvtColor(imgGaussian, cv.COLOR_BGR2HSV)

    #color range, filter white
    lowerBound = np.array([0, 0, 200])
    upperBound = np.array([180, 30, 255])
    
    #binary mask, white pixels (255) represent the color within the defined range, black pixels (0) represent the rest
    mask = cv.inRange(imgHSV, lowerBound, upperBound)
    
    #image filtering, applies the mask to the original image, keeps only the pixels that fall within the color range
    imgFiltered = cv.bitwise_and(img, img, mask=mask)
    
    #store image
    cv.imwrite(f"filteredBlackImg{i+1}.jpg", imgFiltered)

#STEP 4 - reading the text from the image with easyOCR
reader = easyocr.Reader(['es']) #spanish
reader2 = easyocr.Reader(['es']) #spanish

#read text from image
result = reader.readtext("./ImageTextDetector/filteredBlackImg1.jpg")
result2 = reader.readtext("./ImageTextDetector/filteredBlackImg2.jpg")

#print the extracted text with confidence rate
for (bbox, text, prob) in result:
    print(f"Text: {text}, Detection confidence: {prob}")
    
#print the extracted text with confidence rate
for (bbox, text, prob) in result2:
    print(f"Text: {text}, Detection confidence: {prob}")