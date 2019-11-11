import cv2
import numpy as np
from labels import label2color, labelsWithTrainID



def convertToClasses(mask, classes):
    """ This function takes a label-id ground truth mask and converts it to a mask only containing the labels specified in the classes vector.
        A conversion dict that gives the original id for the 'new' classes is also returned.
    """
    IdConverter = dict()
    height, width = mask.shape[:2]
    newmask = -1 * np.ones((height, width)).astype(np.uint8)
    for k, classId in enumerate(classes):
        newmask[mask == classId] = k
        IdConverter[k] = classId
    return newmask, IdConverter


def visualizeMask(mask, classes, IdConverter, rgbimage = None):
    """ This function visualizes the in classes specified classes for a ground truth mask mask.
    """
    height, width = mask.shape[:2]
    outputimage = np.zeros((height, width, 3)).astype(np.uint8)
    for classId in classes:
        outputimage[mask == classId] = label2color[IdConverter[classId]]
    if rgbimage.any():
        outputimage = cv2.cvtColor(outputimage, cv2.COLOR_BGR2HSV)
        rgbimage = cv2.cvtColor(rgbimage, cv2.COLOR_BGR2HSV)
        rgbimage[:,:,:2] = outputimage[:,:,:2]
        rgbimage = cv2.cvtColor(rgbimage, cv2.COLOR_HSV2RGB)
        cv2.imshow("mask", rgbimage)
    else:
        outputimage = cv2.cvtColor(outputimage, cv2.COLOR_HSV2RGB)
        cv2.imshow("mask", outputimage)
        

if __name__ == "__main__":
    testimage = "bremen_000000_000019"
    testImageRGB = cv2.imread(testimage + "_leftImg8bit.png")
    testGT = cv2.imread(testimage + "_gtFine_labelIds.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("GT",testGT)
    classes = list(range(34)) #[7] #list(range(33))# labelsWithTrainID
    newmask, IdConverter = convertToClasses(testGT, classes)
    visualizeMask(newmask, list(range(len(classes))), IdConverter, testImageRGB)
    
    cv2.waitKey(0)