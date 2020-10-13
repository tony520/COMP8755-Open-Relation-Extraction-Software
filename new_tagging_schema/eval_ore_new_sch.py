"""
Evaluate the performance of BiLSTM-CRF with new tagging schema
"""
import string
import nltk
from nltk.corpus import stopwords

stopwords = stopwords.words('english') + list(string.punctuation)
LEXICAL_THRESHOLD = 0.5

"""
Check if list1(list2) is the sublist of list2(list1)
"""
def sublist(lst1, lst2):
    ls1 = [element for element in lst1 if element in lst2]
    ls2 = [element for element in lst2 if element in lst1]
    return ls1 == ls2

"""
Remove stopwords from a sequence
"""
def remove_stopwords(seq):
    return [w for w in seq if w.lower() not in stopwords]

"""
Compare if two sequences are same
"""
def compare_seq(seq1, seq2):
    if len(seq1) != len(seq2):
        return False
    is_similar = True
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            is_similar = False
    return is_similar

"""
Convert a list to a string
"""
def list_to_string(in_list):
    strg = ''
    strg = ' '.join([str(elem) for elem in in_list])
    return strg

"""
Decide if the predicates are matching in true and predicted tagging sequence
"""
def pred_acc_match(t1_arr, t2_arr):
    if len(t1_arr) != len(t2_arr):
        return False
    for i in range(len(t1_arr)):
        if t1_arr[i][0] == 'P' and t2_arr[i][0] == 'O':
            return False
        if t1_arr[i][0] == 'O' and t2_arr[i][0] == 'P':
            return False
    return True

"""
Get arguments from one sentence
"""
def get_arguments(tags, sentence):
    arguments = []
    arg_0, arg_1, arg_2, arg_3, arg_4, arg_5 = [], [], [], [], [], []
    arg_0_idx, arg_1_idx, arg_2_idx, arg_3_idx, arg_4_idx, arg_5_idx = [], [], [], [], [], []
    
    for i in range(len(tags)):
        if tags[i] in 'A0-B' or tags[i] in 'A0-I':
            arg_0.append(sentence[i])
            arg_0_idx.append(i)
        elif tags[i] in 'A1-B' or tags[i] in 'A1-I':
            arg_1.append(sentence[i])
            arg_1_idx.append(i)
        elif tags[i] in 'A2-B' or tags[i] in 'A2-I':
            arg_2.append(sentence[i])
            arg_2_idx.append(i)
        elif tags[i] in 'A3-B' or tags[i] in 'A3-I':
            arg_3.append(sentence[i])
            arg_3_idx.append(i)
        elif tags[i] in 'A4-B' or tags[i] in 'A4-I':
            arg_4.append(sentence[i])
            arg_4_idx.append(i)
        elif tags[i] in 'A5-B' or tags[i] in 'A5-I':
            arg_5.append(sentence[i])
            arg_5_idx.append(i)
    if arg_0:
        a0 = ''
        for i in arg_0_idx:
            a0 += sentence[i] + ' '
        arguments.append(a0.strip())
    if arg_1:
        a1 = ''
        for i in arg_1_idx:
            a1 += sentence[i] + ' '
        arguments.append(a1.strip())
    if arg_2:
        a2 = ''
        for i in arg_2_idx:
            a2 += sentence[i] + ' '
        arguments.append(a2.strip())
    if arg_3:
        a3 = ''
        for i in arg_3_idx:
            a3 += sentence[i] + ' '
        arguments.append(a3.strip())
    if arg_4:
        a4 = ''
        for i in arg_4_idx:
            a4 += sentence[i] + ' '
        arguments.append(a4.strip())
    if arg_5:
        a5 = ''
        for i in arg_5_idx:
            a5 += sentence[i] + ' '
        arguments.append(a5.strip())
    
    return arguments
            

"""
Return whether the actual relations and predicted relations are similar
"""
def predMatch(actual_tags, predict_tags, sentence, ignoreStopwords, ignoreCase):
    # Get the predicate from the actual tags
    pred_actual = []
    pred_actual_idx = []
    for i in range(len(actual_tags)):
        if actual_tags[i] in 'P-B' or actual_tags[i] in 'P-I':
            pred_actual.append(actual_tags[i])
            pred_actual_idx.append(i)
    predicate_actual = []
    for x in pred_actual_idx:
        predicate_actual.append(sentence[x])
        
    # Get the predicate from the predicted tags
    pred_predict = []
    pred_predict_idx = []
    for i in range(len(predict_tags)):
        if predict_tags[i] in 'P-B' or predict_tags[i] in 'P-I':
            pred_predict.append(predict_tags[i])
            pred_predict_idx.append(i)
    
    predicate_predict = []
    if pred_predict:
        for x in pred_predict_idx:
            predicate_predict.append(sentence[x])
        
        s1 = list_to_string(predicate_actual)
        s2 = list_to_string(predicate_predict)
        
        if ignoreCase:
            s1 = s1.lower()
            s2 = s2.lower()
        
        words_s1 = s1.split(' ')
        words_s2 = s2.split(' ')
        
        if ignoreStopwords:
            words_s1 = remove_stopwords(words_s1)
            words_s2 = remove_stopwords(words_s2)
        return (sublist(words_s2, words_s1) or sublist(words_s1, words_s2))

"""
Return the number of predicates in true and predicted tags
"""
def getPredMatch(actual_tags, predict_tags, sentence, ignoreStopwords, ignoreCase):
    # Get the predicate from the actual tags
    pred_actual = []
    pred_actual_idx = []
    for i in range(len(actual_tags)):
        if actual_tags[i] in 'P-B' or actual_tags[i] in 'P-I':
            pred_actual.append(actual_tags[i])
            pred_actual_idx.append(i)
    predicate_actual = []
    for x in pred_actual_idx:
        predicate_actual.append(sentence[x])
        
    # Get the predicate from the predicted tags
    pred_predict = []
    pred_predict_idx = []
    for i in range(len(predict_tags)):
        if predict_tags[i] in 'P-B' or predict_tags[i] in 'P-I':
            pred_predict.append(predict_tags[i])
            pred_predict_idx.append(i)
    
    predicate_predict = []
    if pred_predict:
        for x in pred_predict_idx:
            predicate_predict.append(sentence[x])
        
        s1 = list_to_string(predicate_actual)
        s2 = list_to_string(predicate_predict)
        
        if ignoreCase:
            s1 = s1.lower()
            s2 = s2.lower()
        
        words_s1 = s1.split(' ')
        words_s2 = s2.split(' ')
        
        if ignoreStopwords:
            words_s1 = remove_stopwords(words_s1)
            words_s2 = remove_stopwords(words_s2)
        print("Predicted relations: ", words_s2)
        print("True relations: ", words_s1)
        return min(len(words_s2), len(words_s1)), len(words_s1)
    
"""
Return the number of predicates in validation sentences
"""
def getTruePredicate(actual_tags, sentence, ignoreStopwords, ignoreCase):
    # Get the predicate from the actual tags
    pred_actual = []
    pred_actual_idx = []
    for i in range(len(actual_tags)):
        if actual_tags[i] in 'P-B' or actual_tags[i] in 'P-I':
            pred_actual.append(actual_tags[i])
            pred_actual_idx.append(i)
    predicate_actual = []
    for x in pred_actual_idx:
        predicate_actual.append(sentence[x])
        
    s1 = list_to_string(predicate_actual)

    if ignoreCase:
        s1 = s1.lower()

    words_s1 = s1.split(' ')

    if ignoreStopwords:
        words_s1 = remove_stopwords(words_s1)
    return len(words_s1)


"""
Compute the accuracy of the model
"""
def compute_accuracy(predict_tags, actual_tags):
    assert len(predict_tags) == len(actual_tags)
    correct_predict = 0
    for i in range(len(predict_tags)):
        compared = compare_seq(predict_tags[i], actual_tags[i])
        if compared:
            correct_predict += 1
    accuracy = correct_predict / len(predict_tags)
    return accuracy

"""
Compute the y_pred and y_true for Precision-Recall curve
"""

def compute_PR_values(predict_tags, actual_tags, sentences, ignoreStopwords, ignoreCase):
    assert len(predict_tags) == len(actual_tags) == len(sentences)
    y_pred = []
    y_true = []
    for i in range(len(predict_tags)):
        if predMatch(actual_tags[i], predict_tags[i], sentences[i], ignoreStopwords, ignoreCase):
            getP, getT = getPredMatch(actual_tags[i], predict_tags[i], sentences[i], ignoreStopwords, ignoreCase)
            y_pred.append(getP)
            y_true.append(getT)
        else:
            realT = getTruePredicate(actual_tags[i], sentences[i], ignoreStopwords, ignoreCase)
            y_true.append(realT)
            y_pred.append(0)
    return y_pred, y_true

"""
Compute the predicate matching score
"""
def compute_predMatch(predict_tags, actual_tags, sentences, ignoreStopwords, ignoreCase):
    assert len(predict_tags) == len(actual_tags) == len(sentences)
    match_predicate = 0
    for i in range(len(predict_tags)):
        if predMatch(actual_tags[i], predict_tags[i], sentences[i], ignoreStopwords, ignoreCase):
            match_predicate += 1
            getP, getT = getPredMatch(actual_tags[i], predict_tags[i], sentences[i], ignoreStopwords, ignoreCase)
            y_pred.append(getP)
            y_true.append(getT)
        else:
            realT = getTruePredicate(actual_tags[i], sentences[i], ignoreStopwords, ignoreCase)
            y_true.append(realT)
            y_pred.append(0)
    match_rate = float(match_predicate) / len(predict_tags)
    return match_rate