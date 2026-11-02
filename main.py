#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Naive Bayes implementation.
Maximizes the log likelihood to prevent underflow,
and applies Laplace smoothing to solve the zero observations problem.

API inspired by SciKit-learn.

Sources:
  * https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
  * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/naive_bayes.py
  * https://github.com/ashkonf/HybridNaiveBayes
"""

from datasets import load_loan_defaulters
from feature import ContinuousFeature
from feature import DiscreteFeature
from naive_bayes import NaiveBayes


def main():
    dataset = load_loan_defaulters()
    design_matrix = [row[:-1] for row in dataset]
    target_values = [row[-1] for row in dataset]
    clf = NaiveBayes(extract_features)
    clf.fit(design_matrix, target_values)
    report_training_accuracy(clf, design_matrix, target_values)
    prediction = clf.predict_record([1, 1, 50700])
    negation_word = " not " if prediction == 0.0 else ""
    print("We predict this person will" + negation_word + "default on their loans.")


def report_training_accuracy(clf, design_matrix, target_values):
    """Predict every training record and print how many were correct.

    This makes the demo show whether the classifier actually works,
    instead of only printing one prediction. Note that accuracy on
    the training data is optimistic -- the model has already seen
    these records -- so treat it as a sanity check, not a final grade.
    """
    predictions = clf.predict(design_matrix)
    correct = sum(1 for predicted, actual
                  in zip(predictions, target_values)
                  if predicted == actual)
    total = len(target_values)
    print("Training-set accuracy: {}/{} = {:.1%}".format(
        correct, total, correct / total))
    print("(Optimistic: the model was trained on these same records.)")


def extract_features(feature_vector):
    """Maps a feature vector to whether each feature is continuous or discrete."""
    return [
        DiscreteFeature(feature_vector[0]),
        DiscreteFeature(feature_vector[1]),
        ContinuousFeature(feature_vector[2])
    ]


if __name__ == '__main__':
    main()
