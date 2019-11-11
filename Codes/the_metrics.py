import numpy as np
from predict_images import *



def IOU(Y_true,Y_pred):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IOUs = []
    #classe = int(np.max(Y_true)) + 1
    classe = 9
    for c in range(classe):
        TP = np.sum( (Y_true == c)&(Y_pred==c) )
        FP = np.sum( (Y_true != c)&(Y_pred == c) )
        FN = np.sum( (Y_true == c)&(Y_pred != c)) 
        IOU = TP/float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IOU={:4.3f}".format(c,TP,FP,FN,IOU))
        IOUs.append(IOU)
    mIOU = np.mean(IOUs)
    print("_________________")
    print("Mean IOU is: {:4.3f}".format(mIOU*100))
    
IOU(Y_test,Y_pred)

