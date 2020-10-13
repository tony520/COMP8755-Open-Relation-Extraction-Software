# Transfer Stanovsky dataset
# Load all data
def select_row_data(filename):
    all_sents = open(filename, "r")
    all_data_sentences = (all_sents.read()).split("\n")
    return all_data_sentences

# Load training set
def select_dataset(sentsname, tagsname):
    train_sents = open(sentsname, "r")
    training_sentences = (train_sents.read()).split("\n")
    train_tags = open(tagsname, "r")
    training_tags = (train_tags.read()).split("\n")
    return training_sentences, training_tags

# Stanovsky dataset
all_data_sentences = select_row_data("data/train+test.oie.sents.txt")
training_sentences, training_tags = select_dataset("data/train.oie.sents.txt", "data/train.oie.tags.txt")

# Load training data (Default - Stanovsky)
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
    return training_data

training_data = load_train_dataset(training_sentences, training_tags)
all_data = [all_data_sentences[i].split() for i in range(len(all_data_sentences))]
print("We have %i training data" % (len(training_data)))
print("We have %i data (training + testing)" % len(all_data))

testing_sentences, testing_tags = select_dataset("data/test.oie.sents.txt", "data/test.oie.tags.txt")
testing_data = load_train_dataset(testing_sentences, testing_tags)
print("We have %i testing data" % (len(testing_data)))

with open("src-train-gs.txt", 'w') as f:
    for line in training_sentences:
        if len(line) > 0:
            f.write(line)
            f.write('\n')