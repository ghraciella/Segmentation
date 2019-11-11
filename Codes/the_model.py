# Import necessary items from Keras
from keras.models import Model
from keras.layers import Input, Activation, Dropout, UpSampling2D, Add, add
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers



classes =  9

def FCN8model():
    
    # input
    inputs = Input(shape = (256,256, 3), name = 'input_layer')

    #first conv block w/pool

    conv11 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv11')(inputs)
    conv12 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv12')(conv11)
    pool1 =  MaxPooling2D(pool_size=(2,2), strides=(2,2), name = 'pool1')(conv12)


    #second conv block w/pool

    conv21 = Conv2D(filters = 128, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv21')(pool1)
    conv22 = Conv2D(filters = 128, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv22')(conv21)
    pool2 =  MaxPooling2D(pool_size=(2,2), strides=(2,2), name = 'pool2')(conv22)


    #third conv block w/pool

    conv31 = Conv2D(filters = 256, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv31')(pool2)
    conv32 = Conv2D(filters = 256, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv32')(conv31)
    conv33 = Conv2D(filters = 256, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv33')(conv32)
    pool3 =  MaxPooling2D(pool_size=(2,2), strides=(2,2), name = 'pool3')(conv33)


    #fourth conv block w/pool

    conv41 = Conv2D(filters = 512, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv41')(pool3)
    conv42 = Conv2D(filters = 512, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv42')(conv41)
    conv43 = Conv2D(filters = 512, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv43')(conv42)
    pool4 =  MaxPooling2D(pool_size=(2,2), strides=(2,2), name = 'pool4')(conv43)


    #fifth conv block w/pool

    conv51 = Conv2D(filters = 512, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv51')(pool4)
    conv52 = Conv2D(filters = 512, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv52')(conv51)
    conv53 = Conv2D(filters = 512, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv53')(conv52)
    pool5 =  MaxPooling2D(pool_size=(2,2), strides=(2,2), name = 'pool5')(conv53)


    #sixth and seventh conv block w/dropout

    conv6 = Conv2D(filters =4096 , kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv6')(pool5)
    dropout1 = Dropout(0.85, name = 'dropout1')(conv6)

    conv7 = Conv2D(filters = 4096, kernel_size = 3, activation = 'relu', 
                                        strides = (1,1), padding = 'same', name = 'conv7')(dropout1)

    dropout2 = Dropout(0.85, name = 'dropout2') (conv7)

    #8th conv layer withclassification scores for each class

    conv_score = Conv2D(filters = classes, kernel_size = 1, activation = 'relu', name = 'conv8')(dropout2)
    
    
    encoder = Model(inputs , conv_score)

    out_put1 = encoder.layers[-1].output
    out_size1 = encoder.layers[-1].output_shape[2]
    #print(out_size1)
    

    #first deconvolution layer

    deconv1 = Conv2DTranspose(filters = 512, kernel_size = 4, strides = (2,2),padding = 'valid' , name = 'deconv1')(out_put1)
    
    test = Model(inputs, deconv1)
    out_put2 = test.layers[-1].output
    out_size2 = test.layers[-1].output_shape[2]
    
    #print(out_size2)

    #to get correct crop_value based on image size
    crop_value = out_size2 - 2*out_size1
    #print(crop_value)

    Crop1 = Cropping2D(cropping = ((0,crop_value),(0,crop_value)), name = 'Crop1')(deconv1)

    #adds output pool4 and cropped upsampled last layer to give a segmentation map
    segmap1 = Add()([pool4, Crop1])


    for_up = Conv2D(filters = classes, kernel_size = 1, activation = 'relu', name = 'for_up')(segmap1)
    
    
    test1 = Model(inputs , for_up)

    out_put11 = test1.layers[-1].output
    out_size11 = test1.layers[-1].output_shape[2]
    #print(out_size11)
    #second deconvolution layer

    deconv2 = Conv2DTranspose(filters = 256, kernel_size = 4, strides = (2,2),padding = 'valid' , name = 'deconv2')(segmap1)

    test2 = Model(inputs, deconv2)

    out_put22 = test2.layers[-1].output
    out_size22 = test2.layers[-1].output_shape[2]
    #print(out_size22)

    #to get correct crop_value based on image size
    crop_value2 = out_size22 - 2*out_size11
    #print(crop_value2)

    Crop2 = Cropping2D(cropping = ((0,crop_value2),(0,crop_value2)), name = 'Crop2')(deconv2)

    #adds output pool3 and cropped upsampled last layer to give a segmentation map
    segmap2 = Add()([pool3, Crop2])


    #third deconvolution layer

    deconv3 = Conv2DTranspose(filters = classes, kernel_size = 16, strides = (8,8),padding = 'same' , name = 'deconv3')(segmap2)


    #softmax activation

    soft_preds = Activation('softmax', name = 'soft_preds')(deconv3)

    model = Model(inputs = inputs, outputs = soft_preds)
    model.summary()

    return model


#the_model = FCN8model()


