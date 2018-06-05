import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from keras.callbacks import ModelCheckpoint

train_df = np.load('array_train_cup.npy')
print(train_df.shape)

y_train= train_df[:, :,[1]]
x_train= train_df[:, :,[0,2,3,4,5,6,7,8,9]]

train_data, remaining_data, train_labels, remaining_labels = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(remaining_data, remaining_labels, test_size=0.30, random_state=42)

def euclidean_distance_loss(y_true, y_pred):
    zero = tf.constant(0, dtype='float32',name=None)
    where = tf.not_equal(y_true, zero)
    omit_zeros = tf.boolean_mask(y_true,where)
    indices = tf.where(where)
    result = tf.gather_nd(y_pred, indices)
    return K.sqrt(K.sum(K.square(omit_zeros -(result)), axis=-1))

model = Sequential()
model.add(LSTM(16, input_shape=(1099,9),activation='tanh',recurrent_activation='hard_sigmoid',return_sequences=True))
model.add(LSTM(16, activation='tanh',recurrent_activation='hard_sigmoid',return_sequences=True))
model.add(LSTM(16, activation='tanh',recurrent_activation='hard_sigmoid',return_sequences=True))
model.add(LSTM(16, activation='tanh',recurrent_activation='hard_sigmoid', return_sequences=True))
model.add(Dense(1, activation='relu'))
model.add(LSTM(16, activation='tanh',recurrent_activation='hard_sigmoid', return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu')) 
model.compile(loss=euclidean_distance_loss, optimizer=Adam(lr=0.0001))
print(model.summary())

checkpointer_model = ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=True, mode='min')
checkpointer_weights = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')

model.fit(train_data, train_labels, nb_epoch=2, batch_size=16, validation_data=(val_data, val_labels), callbacks=[checkpointer_model, checkpointer_weights])

predWeight = model.predict(test_data,batch_size=1)

s = np.linspace(10, 1099, num=1099)
fig, ax = plt.subplots( nrows=1, ncols=1 )  
for x in range(0, 16):
    plt.plot(s, test_labels[x+30,:,0],'C1', label='Actual')
    plt.plot(s, predWeight[x+30,:,0],'C2', label='Prediction')
    plt.legend()
    fig.savefig('graph'+str(x)+'.png')   
    fig.clf()
plt.close(fig)    
