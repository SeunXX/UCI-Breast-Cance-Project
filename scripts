# import the several packages that will be needed

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# import data into panda 
breast_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
breast_df.columns = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion", "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"]

# Locate missing data using descriptive statistics
breast_df.describe()
breast_df = breast_df.replace('?', '0')
breast_df1 = breast_df.iloc[:, 1:-1]

# Define labels
features= ['CodeNumber', 'ClumpThickness', 'UniformityCellSize', 'UniformityCellShape','MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses']
X = breast_df[features]
y = breast_df.loc[:, "CancerType"]

# Split dataset, fit and train model, predict
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.5, random_state = 100)

cancer_forestModel = RandomForestClassifier(criterion = 'gini')
cancer_forestModel.fit(train_X, train_y)
y_prediction = cancer_forestModel.predict(val_X)

# Accuracy of model
accuracy = accuracy_score(val_y,y_prediction)*100
print(round(accuracy), '%')

from sklearn import metrics
from sklearn.metrics import mean_absolute_error
mean_absolute_error(val_y, y_prediction)

# Predict the cancer type: Benign = 2, Malignant = 4
Id = val_X['CodeNumber']
Prediction = pd.DataFrame({'CodeNum': Id, 'Obs_Cancer_Type' : val_y, 'Predicted_Cancer_Type' : y_prediction})
