import numpy as np
import pandas as pd
from metrics import *
import matplotlib
import matplotlib.pyplot as plt

"""
Mapping the predicted sequence to a tagging sequence
"""
def mapping_tags(predict_sequence):
    tagging_seq = [ix_to_tag[t] for t in predict_sequence]
    return tagging_seq

"""
Get the predicted tags in length = step_length
"""
def getPredictY(model, word_to_ix, step_length, test_sents):
    predict_tags = []
    for i in range(step_length):
        splitted_sents = test_sents[i]
        prepared_sents = prepare_sequence(splitted_sents, word_to_ix)
        predict_tagging_sequence = model(prepared_sents)[1]
        predict_tags.append(mapping_tags(predict_tagging_sequence))
    
    return predict_tags

"""
Get the true tags in length = step_length
"""
def getTrueY(step_length, test_tags):
    true_tags = test_tags[0:step_length]
    return true_tags

"""
Get the sentence in length = step_length
"""
def getSentInStep(step_length, test_sents):
    sents = test_data_sents[0:step_length]
    return sents

"""
Generate the true predicates and predicted predicates (y_true and y_pred)
"""
def compute_predMatch_pr(model, word_to_ix, step_length, test_sents, test_tags, ignoreStopwords, ignoreCase):
    predict_tags = getPredictY(model, word_to_ix, step_length, test_sents)
    true_tags = getTrueY(step_length, test_tags)
    sents = getSentInStep(step_length, test_sents)
    
    assert len(predict_tags) == len(true_tags) == len(sents)
    
    yp, yt = [], []
    for i in range(len(predict_tags)):
        if predMatch(true_tags[i], predict_tags[i], sents[i], ignoreStopwords, ignoreCase):
            getP, getT = getPredMatch(true_tags[i], predict_tags[i], sents[i], ignoreStopwords, ignoreCase)
            yp.append(getP)
            yt.append(getT)
        else:
            realT = getTruePredicate(true_tags[i], sents[i], ignoreStopwords, ignoreCase)
            yt.append(realT)
            yp.append(0)
    return yp, yt

"""
Generate precisions and recalls
"""
def generate_precision_recall_values(model, word_to_ix, step_length, test_data_sents, test_data_tags, ignoreStopwords, ignoreCase):
	prec_arr = []
	rec_arr = []
	i = step_length
	while i < len(test_data_sents):
		yp, yt = compute_predMatch_pr(model, word_to_ix, i, test_data_sents, test_data_tags, ignoreStopwords = True, ignoreCase = True)
		temp_report = metrics.classification_report(yt, yp, digits = 6)
		prec_arr.append(temp_report.strip().split('\n')[-1].strip().split()[2])
		rec_arr.append(temp_report.strip().split('\n')[-1].strip().split()[3])
		i += step_length

	df = pd.DataFrame({'precision': prec_arr, 'recall': rec_arr})
	df.to_csv('BiLSTM-CRF.dat', index = False)

"""
Plot precision-recall curve (experiment result and baseline result)
"""
def plot_pr_curve(exper_result, baseline):
	df = pd.read_csv(exper_result)
	recall_ord = np.array(df['recall'].to_list())
	precision_ord = np.array(df['precision'].to_list())

	precision_gs = []
	recall_gs = []
	metrics_gs = [i.strip().split() for i in open(baseline).readlines()]
	for i in range(1, len(metrics_gs)):
	    precision_gs.append(float(metrics_gs[i][0]))
	    recall_gs.append(float(metrics_gs[i][1]))

	df_gs = pd.DataFrame({'precision_rnnoie': precision_gs, 'recall_rnnoie': recall_gs})
	df_gs.to_csv('baseline.dat', index = False)
	plt.plot(recall_ord, precision_ord, label = "BiLSTM-CRF")
	plt.plot(recall_gs, precision_gs, label = "RNNOIE-AW (Stanovsky)")
	plt.axis([0, 1, 0, 1])
	plt.title('Precision-Recall curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend()
	plt.show()