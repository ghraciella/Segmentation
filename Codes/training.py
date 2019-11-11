from keras import optimizers
from keras import losses
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import numpy as np
import datetime
from time import time
from the_model import *
#from imgloadgen import *


'''
To Start TensorBoard 
In notebooks, use the %tensorboard line magic. 
On the command line, run the same command without "%".

%tensorboard --logdir logs/fit

'''

def loss_functions(loss='categorical_crossentropy'):

    """
    put documentation
    """

    if loss == 'categorical_crossentropy':
        loss = losses.categorical_crossentropy
    if loss == 'sparse_categorical_crossentropy':
        loss = losses.sparse_categorical_crossentropy     

    return loss


def the_optimizers(lr=1E-2, optim = 'sgd'):

    """
    put documentation
    """
    if optim == 'sgd':
        optim = optimizers.SGD(lr=1E-2, momentum=0.0, nesterov=False)

    if optim == 'sgdmn':
        optim = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)

    if optim == 'adam':
        optim = optimizers.SGD(lr=1E-2, beta_1=0.9, beta_2=0.999, amsgrad=False)

    if optim == 'rmsprop':
        optim = optimizers.RMSprop(lr=1E-2, rho = 0.9)

    if optim == 'nadam':
        optim = optimizers.SGD(lr=2E-2, beta_1=0.9, beta_2=0.999)

    return optim


def training(model, optimizer, X_train, Y_train, X_val, Y_val):
    """
    put documentation
    """

    #the_model.load_weights('modelweights/mymodelweights.best.h5')

    the_model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['accuracy'])

    #log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
            ModelCheckpoint(
                #filepath='mymodel_{epoch}.h5',
                # save the model to path and the currentcheckpoint changes
                #based on if the val loss is improved
                #and save best only is true
                filepath = 'modelweights/mymodelweights.best.h5', 
                #save the model weights to the same file, 
                #if and only if  the validation accuracy improves.
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss',
                verbose=1),
            #TensorBoard(log_dir=log_dir,
            #    histogram_freq=1),    
            EarlyStopping(monitor='val_loss',
                patience=50,
                verbose=1,
                mode='auto'), 
            #ReduceLROnPlateau(monitor='val_loss', 
            #    factor=0.2,
            #    patience=5, 
            #    min_lr=0.001)       
    ]

    start_time = time()
    train_hist = the_model.fit(X_train,Y_train,
                    validation_data=(X_val,Y_val),
                    batch_size=32,
                    epochs=500,
                    verbose=1,
                    callbacks=callbacks)

    train_time = time() - start_time
    print(train_time)                

    the_model.save('savedmodels/mymodel.h5')    
                        
    return train_hist




if __name__ == '__main__':

    npypath = 'SavedNpy'
    X_train, Y_train = np.load(npypath + '/X_train_data.npy'), np.load(npypath + '/Y_train_data.npy')
    X_val, Y_val = np.load(npypath + '/X_val_data.npy'), np.load(npypath + '/Y_val_data.npy')

    the_model = FCN8model()

    loss = loss_functions(loss='categorical_crossentropy')

    optimizer = the_optimizers(lr=1E-2, optim = 'sgd')    

    train_datahist = training(the_model, optimizer, X_train, Y_train, X_val, Y_val)

    score = the_model.evaluate(X_val, Y_val, verbose=2)
    print("%s: %.2f%%" % (the_model.metrics_names[1], score[1]*100))



