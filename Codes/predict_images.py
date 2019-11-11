from keras.models import load_model  
from training import * 

the_model=load_model('mymodel.h5')


Y_pred = the_model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=3)
Y_test = np.argmax(Y_test, axis=3)
print(Y_test.shape,Y_pred.shape)


