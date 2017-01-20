#!/usr/bin/python

from sklearn.model_selection import StratifiedShuffleSplit

def test_classifier(clf, features, labels):
  cv = StratifiedShuffleSplit(20, test_size=0.25, random_state = 42)
  true_negatives = 0
  false_negatives = 0
  true_positives = 0
  false_positives = 0
  for train_index, test_index in cv.split(features, labels):
    features_train, features_test = features[train_index], features[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
      if prediction == 0 and truth == 0:
        true_negatives += 1
      elif prediction == 0 and truth == 1:
        false_negatives += 1
      elif prediction == 1 and truth == 0:
        false_positives += 1
      elif prediction == 1 and truth == 1:
        true_positives += 1
      else:
        print "Warning: Found a predicted label not == 0 or 1."
        print "All predictions should take value 0 or 1."
        print "Evaluating performance for processed predictions:"
        break
  try:
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = float((true_positives + true_negatives))/total_predictions
    print 'accuracy of the classifier is ', accuracy, ' on the test data'
  except:
    print "Got a divide by zero when trying out:", clf
    print "Precision or recall may be undefined due to a lack of true positive predicitons."

