
import itertools
import os
import os.path as path

import numpy as np
from cv2 import cv2
from numpy import asarray

height = 256
width = 256
classes = 9
#batch_size = 32

#define paths to train, val and test images and segmentation ground truth
#dir_data = '../Datasets/cityscapes'
dir_data = path.abspath(path.join(os.getcwd(),"../../Datasets/cityscapes/AllbyGrac"))
#print(dir_data)


dir_img = dir_data + '/Joint_trainim'
dir_seg = dir_data + '/Joint_traingt'

# print(dir_img)
# print(dir_seg)

val_dir_img = dir_data + '/Joint_valim'
val_dir_seg = dir_data + '/Joint_valgt'


test_dir_img = dir_data + '/Joint_testim'
test_dir_seg = dir_data + '/Joint_testgt'



#define the needed functions

def normalize(image, norm = "smean" ):
    """
    put documentation
    """

    if norm == "gray_norm":
        image = image.astype('float32')
        image = image/255.0

    elif norm == "smean":
        image = image.astype('float32')
        image[:,:,0] -= 103.939
        image[:,:,1] -= 116.779
        image[:,:,2] -= 123.68
        
    elif norm == 'pixelmean':
        image = asarray(image)
        image =image.astype('float32')
        mean, std = image.mean(axis=(0,1), dtype = 'float64'), image.std(axis=(0,1), dtype = 'float64')
        #print('mean: %s, std: %s' %(mean,std))
        image = (image - mean)/std
    
    return image


def segimages( img_path, height, width ):

    """
    put documentation
    """    

    images = cv2.imread(img_path, 1)
    assert not isinstance(images,type(None)), 'image not found'
    images = cv2.resize(images, ( height, width ))
    images = normalize(images, norm = "pixelmean")

    return images

def segmasks( gt_path, classes , height, width ):

    """
    put documentation
    """

    seg_labels = np.zeros((height, width, classes ))
    images = cv2.imread(gt_path, 1)
    assert not isinstance(images,type(None)), 'segmentation label not found'
    images = cv2.resize(images, (height, width ))
    images = images[:, : , 0]

    for c in range(classes):
        seg_labels[:, :, c] = (images == c).astype(int)
    return seg_labels
    

def datagenerator(image_dir, gt_dir, classes, height, width, alte = 'train'):

    """
    put documentation
    """

    ## make generator which yields image and masks for loading segmentation data

    # image_dir = sorted(os.listdir(dir_img))
    # gt_dir = sorted(os.listdir(dir_seg))

    if alte == 'train':
        image_dir = sorted(os.listdir(dir_img))
        gt_dir = sorted(os.listdir(dir_seg))

    if alte == 'val':
        image_dir = sorted(os.listdir(val_dir_img))
        gt_dir = sorted(os.listdir(val_dir_seg))

    if alte == 'test':
        image_dir = sorted(os.listdir(test_dir_img))
        gt_dir = sorted(os.listdir(test_dir_seg))

    the_data = itertools.cycle(zip(image_dir, gt_dir))
    while True:
        the_images = []    
        the_gtmask = []  
        batch_size = len(image_dir)
        for _ in range(batch_size):
            im, seg = next(the_data)
            # the_images.append(segimages(dir_img + os.sep + im, height, width))
            # the_gtmask.append(segmasks(dir_seg + os.sep + seg, classes, height, width ))

            if alte == 'train':
                the_images.append(segimages(dir_img + os.sep + im, height, width))
                the_gtmask.append(segmasks(dir_seg + os.sep + seg, classes, height, width ))

            if alte == 'val':
                the_images.append(segimages(val_dir_img + os.sep + im, height, width))
                the_gtmask.append(segmasks(val_dir_seg + os.sep + seg, classes, height, width ))

            if alte == 'test':
                the_images.append(segimages(test_dir_img + os.sep + im, height, width))
                the_gtmask.append(segmasks(test_dir_seg + os.sep + seg, classes, height, width ))

        yield np.array(the_images), np.array(the_gtmask)
        

# def saveas_npy(data = X_train, filename = 'X_train', npydir = 'SavedNPY'):
#     """
#     put documentation
#     file name:  should be string
#     """
#     file_name = filename + '_data.npy'

#     if os.path.isfile(file_name):
#         print('file exists, load previous data')
#     else:
#         print('file does not exist, starting fresh')


#     if len(data) % 1000 == 0:
#         print(len(data))
#         np.save(os.path.join(npydir + os.sep + file_name), data)
            

def saveas_npy(data, filename, npydir = 'SavedNpy'):
    """
    put documentation
    file name:  should be string
    """
    file_name = filename + '_data.npy'

    if os.path.isfile(file_name):
        print('file exists, load previous data')
        np.load(file_name)
   
    else:
        print('file does not exist, start fresh')
        #np.save(file_name, data)
        print(len(data))
        np.save(os.path.join(npydir + os.sep + file_name), data)


    return            



if __name__ == '__main__':


    train_data =  datagenerator(dir_img, dir_seg, classes, height, width, alte = 'train')
    X_train, Y_train = next(iter(train_data))

    saveas_npy(X_train, 'X_train', 'SavedNpy')
    saveas_npy(Y_train, 'Y_train', 'SavedNpy')


    val_data =  datagenerator(val_dir_img, val_dir_seg, classes, height, width, alte = 'val')
    X_val, Y_val = next(iter(val_data))

    saveas_npy(X_val, 'X_val', 'SavedNpy')
    saveas_npy(Y_val, 'Y_val', 'SavedNpy')

    # test_data =  datagenerator(test_dir_img, test_dir_seg, classes, height, width, alte = 'test')
    # X_test, Y_test = next(iter(test_data))
    # saveas_npy(X_test, 'X_test', 'SavedNpy')
    # saveas_npy(Y_test, 'Y_test', 'SavedNpy')
  





