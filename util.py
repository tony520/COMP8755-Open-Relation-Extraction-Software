"""
Convert a list to a string
"""

def list_to_string(in_list):
    strg = ''
    strg = ' '.join([str(elem) for elem in in_list])
    return strg

"""
Load training data and all data
"""
def select_row_data(filename):
    all_sents = open(filename, "r")
    all_data_sentences = (all_sents.read()).split("\n")
    return all_data_sentences

def select_dataset(sentsname, tagsname):
    train_sents = open(sentsname, "r")
    training_sentences = (train_sents.read()).split("\n")
    train_tags = open(tagsname, "r")
    training_tags = (train_tags.read()).split("\n")
    return training_sentences, training_tags

"""
Load training data from raw data files
"""
def load_train_dataset(training_sentences, training_tags):
    assert len(training_sentences) == len(training_tags)
    train_s = []
    train_t = []
    for i in range(len(training_sentences)):
        if len(training_sentences[i]) > 0:
            train_s.append(training_sentences[i])
            train_t.append(training_tags[i])
        else:
            continue
    
    assert len(train_s) == len(train_t)
    training_data = [(train_s[i].split(), train_t[i].split()) for i in range(len(train_t))]
    print("We have %s training data" % (len(training_data)))
    return training_data

"""
Load validating data from raw data files
"""
def load_validation_dataset(validating_sentences, validating_tags):
    assert len(validating_sentences) == len(validating_tags)
    validation_s = []
    validation_t = []
    for i in range(len(validating_sentences)):
        if len(validating_sentences[i]) > 0:
            validation_s.append(validating_sentences[i])
            validation_t.append(validating_tags[i])
        else:
            continue
    
    assert len(validation_s) == len(validation_t)
    validating_data_sents = [validation_s[i].split() for i in range(len(validation_s))]
    validating_data_tags = [validation_t[i].split() for i in range(len(validation_t))]
    print("We have %s validation data" % (len(validating_data_sents)))
    return validating_data_sents, validating_data_tags

"""
Mapping the predicted sequence to a tagging sequence
"""
def mapping_tags(predict_sequence):
    tagging_seq = [ix_to_tag[t] for t in predict_sequence]
    return tagging_seq