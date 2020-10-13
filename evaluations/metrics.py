"""
Generate metrics of evaluation
"""
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

stopwords = stopwords.words('english') + list(string.punctuation)
LEXICAL_THRESHOLD = 0.5

"""
Generate the confusion matrix of sequence tagging task (multiple classification metrics)
"""
def generate_confusion_matrix(y_true, y_pred):
	return metrics.confusion_matrix(y_true, y_pred)

"""
Generate the classification report of sequence tagging task
"""
def generate_classification_report(y_true, y_pred):
	return metrics.classification_report(y_true, y_pred)

"""
Generate AUC of ROC curve
"""
def generate_auc_roc_curve(y_true, y_pred):
	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
	return metrics.auc(fpr, tpr)

"""
Return weighted average precision
"""
def generate_precision(y_true, y_pred):
	report = generate_classification_report(y_true, y_pred)
	precision = report.strip().split('\n')[-1].strip().split()[2]
	return precision

"""
Return weighted average recall
"""
def generate_recall(y_true, y_pred):
	report = generate_classification_report(y_true, y_pred)
	recall = report.strip().split('\n')[-1].strip().split()[3]
	return recall

"""
Return weighted average F1-score
"""
def generate_f1_score(y_true, y_pred):
	report = generate_classification_report(y_true, y_pred)
	f1_score = report.strip().split('\n')[-1].strip().split()[4]
	return f1_score