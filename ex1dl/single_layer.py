#Ex1.single_layer

import pandas as pd
from sklearn.linear_model import Perceptron
if __name__ == "__main__":
	db=pd.read_csv('diabetes.csv').values
	x = db[:, 0:8]
	y = db[:, 8]
	model=Perceptron(random_state=1)
	model.fit(x,y)
	print("%0.3f" % model.score(x,y))
