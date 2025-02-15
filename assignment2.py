# Import pandas and our model
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

# Load training data, then separate x and y variables
trainData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
y = trainData['meal']
x = trainData.drop(['meal','id','DateTime'], axis=1)

# Create the model and fit it
model = DecisionTreeClassifier(max_depth=5)
modelFit = model.fit(x, y)

# Load test data, then separate x and y variables
testData = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
y = testData['meal']
x = testData.drop(['meal','id','DateTime'], axis=1)

# Test our model using the testing data
pred = model.predict(x)