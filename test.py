import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

input_df = np.load('array_test_input_cup.npy')
target_df = np.load('array_test_target_cup.npy')

padded_x = pad_sequences(input_df, padding='post', maxlen=1099, dtype='float64')
padded_y = pad_sequences(target_df, padding='post', maxlen=1099, dtype='float64')

model = load_model('my_model.h5')

predWeight = model.predict(padded_x,batch_size=16)
scores = model.evaluate(padded_x, padded_y, batch_size=16)
print(scores)

s = np.linspace(1, 1099, num=1099)
fig, ax = plt.subplots( nrows=1, ncols=1 )  

plt.plot(s, padded_y[286,:,0],'C1', label='Actual')
plt.plot(s, predWeight[286,:,0],'C2', label='Prediction')
plt.legend()
fig.savefig('graph3'+'.png')   
fig.clf()
	
plt.plot(s, padded_y[18,:,0],'C1', label='Actual')
plt.plot(s, predWeight[18,:,0],'C2', label='Prediction')
plt.legend()
fig.savefig('graph4'+'.png')   
fig.clf()
	
plt.plot(s, padded_y[10,:,0],'C1', label='Actual')
plt.plot(s, predWeight[10,:,0],'C2', label='Prediction')
plt.legend()
fig.savefig('graph5'+'.png')   
fig.clf()

plt.plot(s, padded_y[171,:,0],'C1', label='Actual')
plt.plot(s, predWeight[171,:,0],'C2', label='Prediction')
plt.legend()
fig.savefig('graph6'+'.png')   
fig.clf()
	
plt.plot(s, padded_y[267,:,0],'C1', label='Actual')
plt.plot(s, predWeight[267,:,0],'C2', label='Prediction')
plt.legend()
fig.savefig('graph7'+'.png')   
fig.clf()
	
plt.plot(s, padded_y[203,:,0],'C1', label='Actual')
plt.plot(s, predWeight[203,:,0],'C2', label='Prediction')
plt.legend()
fig.savefig('graph8'+'.png')   
fig.clf()
plt.close(fig)