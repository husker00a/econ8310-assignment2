import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load training data, then separate x and y variables
trainData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
y = trainData['meal']
x = trainData.drop(['meal','id','DateTime'], axis=1)

model = XGBClassifier(n_estimators=50, max_depth=3,learning_rate=0.5, objective='multi:softmax')
# Fit to our training split
modelFit = model.fit(x, y)