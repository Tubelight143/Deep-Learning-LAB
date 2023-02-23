#Ex1.single_layer1

import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import hard_sigmoid
if __name__ == "__main__":
	db=pd.read_csv('diabetes.csv').values
	x = db[:, 0:8]
	y = db[:, 8]
	model = Sequential()
	model.add(Dense(1, input_shape=(8,), activation=hard_sigmoid, kernel_initializer='glorot_uniform'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(x, y, epochs=225, batch_size=25, verbose=1, validation_split=0.2)
	_, accuracy = model.evaluate(x, y)
	print("%0.3f" % accuracy)
