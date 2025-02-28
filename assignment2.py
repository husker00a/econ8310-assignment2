import pandas as pd
import numpy as np

trainData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

# Upper case before split, lower case after
Y = trainData['meal']
# make sure you drop a column with the axis=1 argument
X = trainData.drop(['meal','id','DateTime'], axis=1) 

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=50,max_depth=4,learning_rate=0.5,objective="binary:logistic")

# Fit to our training split
modelFit = model.fit(X, Y)

#Test Data
testData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
xt = testData.drop(['meal','id','DateTime'], axis=1) 

# Make predictions based on the testing x values
pred = model.predict(xt)
