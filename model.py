#!/usr/bin/python

import csv as csv
import numpy as np
#import matplotlib.pyplot as plt
import os
import re
from tester import test_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

cwd = os.getcwd()
trainFile = cwd + '/trainData/train.csv'
testFile = cwd + '/testData/test.csv'
csv_file_object = csv.reader(open(trainFile, 'rb'))
header = csv_file_object.next()
data = []

features_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
features_indices = [2, 4, 6, 7, 9]

# Ceate 2 classes for the fare:
# 1. 0 - 100
# 2. > 100

for row in csv_file_object:
  # Modify the data according to the format I want
  if (row[2] != '' and row[4] != '' and row[6] != '' and row[7] != '' and row[9] != ''):
    row[4] = 1 if (row[4] == 'female') else 2
    ticket_fare = float(row[9])
    row[9] = 1 if (ticket_fare >= 0.0 and ticket_fare < 100) else 2
    data.append(row[0:])

data = np.array(data)

# Modify the data according to the format I want
features = data[0:, features_indices]
labels = data[0:, 1].astype(np.int32)
#clf = RandomForestClassifier(n_estimators=25, min_samples_split=20, min_samples_leaf=10)
clf = svm.SVC()
test_classifier(clf, features, labels)


# First, read in test.csv
test_file = open(testFile, 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()
testData = []
pIdWithAllInfo = []

# Also open the a new file so I can write to it. Call it something descriptive
# Finally, loop through each row in the train file, and look in column index [3] (which is 'Sex')
# Write out the PassengerId, and my prediction.

predictions_file = open("gendermodel.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"]) # write the column headers

test_features_list = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
test_features_indices = [1, 3, 5, 6, 8]

# Ceate 2 classes for the fare:
# 1. 0 - 100
# 2. > 100

# I can't just drop the rows, which don't have all the features I need.
# I should come up with a strategy to fill those rows with data
# if PClass is missing, we can add an extra class. Class 0
# if Sex is missing, we can just assume it is a female
# If SibSp is missing, assume 0
# If Parch is missing, assume 0
# If Fare is missing, assume 0
for row in test_file_object:
  # Modify the data according to the format I want
  
  if(row[1] == ''):
    row[1] = '0'
  if(row[3] == ''):
    row[3] = 'female'
  if(row[5] == ''):
    row[5] = '0'
  if(row[6] == ''):
    row[6] = '0'
  if (row[8] == ''):
    row[8] = '0'
  row[3] = 1 if (row[3] == 'female') else 2
  #if (row[1] != '' and row[3] != '' and row[5] != '' and row[6] != '' and row[8] != ''):
  ticket_fare = float(row[8])
  row[8] = 1 if (ticket_fare >= 0.0 and ticket_fare < 100) else 2
  pIdWithAllInfo.append(row[0])
  testData.append(row[0:])

testData = np.array(testData)
test_features = testData[0:, test_features_indices]
clf_predictions = clf.predict(test_features)
for pid, survive in zip(pIdWithAllInfo, clf_predictions):
  predictions_file_object.writerow([pid, survive])
  #print pid, ':', survive
test_file.close()                       # Close out the files.
predictions_file.close()


# Now I have to split the data into train and test

#plt.xlim(0.0, 2.0)
#plt.ylim(0.0, 3.0)
#plt.scatter(passenger_survival, passenger_classes, color = "b")
#plt.show()

# Explore the training data. Look into it in the following way:
#   - see the features
#   - see the output
#   - see if the features have missing values
#   - see if the missing values can be predicted somehow.
# Split the train data in test and train data.
# Choose some features for prediction model.
# Choose the prediction algorithm.
# Train the algorithm on the training data.
# Use the trained algorithm to predict the output on the section of test data taken from the training data.
# Check the precision of the algorithm on this test data output.
# Repeat the process till desired precision is achieved. ~ 80%
# Run the final prediction algorithm on test data and store the output.