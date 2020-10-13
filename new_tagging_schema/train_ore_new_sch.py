import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle as pickle

from models.model import *
from util import *
from evaluations.evaluations import *
from evaluations.metrics import *
import matplotlib
import matplotlib.pyplot as plt


# Load vocabulary and word_to_ix
words = pickle.load(open(f'./pretrained_word_embeddings/GloVe/6B.50_words.pkl', 'rb'))
word_to_ix = pickle.load(open(f'./pretrained_word_embeddings/GloVe/6B.50_word_idx.pkl', 'rb'))
weights_matrix = pickle.load(open(f'./pretrained_word_embeddings/GloVe/6B.50_weights_matrix.pkl', 'rb'))

# Load training data
training_sentences, training_tags = select_dataset("./data/train.sents.txt", "./data/train.tags.txt")
training_data = load_train_dataset(training_sentences, training_tags)
print("We have %i training data" % (len(training_data)))

# Set parameters
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 50
HIDDEN_DIM = 10
POS_EMBEDDING_DIM = 5

# New tagging schema
tag_to_ix = {"P0-B": 0, "P0-I": 1, "P1-B": 2, "P1-I": 3, "P2-B": 4, "P2-I": 5, "P3-B": 6, "P3-I": 7, "P4-B": 8, "P4-I": 9, "P5-B": 10, "P5-I": 11, "P6-B": 12, "P6-I": 13, "P7-B": 14, "P7-I": 15, "P8-B": 16, "P8-I": 17, "P9-B": 18, "P9-I": 19, "O": 20, START_TAG: 21, STOP_TAG: 22}
ix_to_tag = {0: "P0-B", 1: "P0-I", 2: "P1-B", 3: "P1-I", 4: "P2-B", 5: "P2-I", 6: "P3-B", 7: "P3-I", 8: "P4-B", 9: "P4-I", 10: "P5-B", 11: "P5-I", 12: "P6-B", 13: "P6-I", 14: "P7-B", 15: "P7-I", 16: "P8-B", 17: "P8-I", 18: "P9-B", 19: "P9-I", 20: "O", 21: START_TAG, 22: STOP_TAG}

# Training the model
model = BiLSTM_CRF(len(words), tag_to_ix, weights_matrix, EMBEDDING_DIM, HIDDEN_DIM, POS_EMBEDDING_DIM, START_TAG, STOP_TAG)
optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype = torch.long)
    print(model(precheck_sent))

mean_cost = []

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(50):
    epoch_costs = []
    
    print("Starting epoch %i..." % (epoch))
    for sentence, tags in training_data:
        # 1. Clear the accumulate gradients
        model.zero_grad()
        
        # 2. Get inputs and transfer to tensor of index
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype = torch.long)
        
        # 3. Run the forward pass
        loss = model.neg_log_likelihood(sentence_in, targets)
        epoch_costs.append(loss.data.numpy())
        
        # 4. Compute loss, gradients, update parameters
        loss.backward()
        optimizer.step()
    mean_cost.append(np.mean(epoch_costs))
    print("Epoch %i, cost average: %s" % (epoch, np.mean(epoch_costs)))

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(mapping_tags(model(precheck_sent)))
    
print("Loss in 50 epochs: ", mean_cost)
epoch_num = [i for i in range(50)]
plt.plot(epoch_num, mean_cost)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Load validation data
val_sents = open("./data/val.sents.txt", "r")
validating_sentences = (test_sents.read()).split("\n")
val_tags = open("./data/val.tags.txt", "r")
validating_tags = (test_tags.read()).split("\n")

val_data_sents, val_data_tags = load_validation_dataset(validating_sentences, validating_tags)

# Get prediction results over validation data
predict_tags = []
for i in range(len(val_data_sents)):
    splitted_sents = val_data_sents[i]
    prepared_sents = prepare_sequence(splitted_sents, word_to_ix)
    predict_tagging_sequence = model(prepared_sents)[1]
    predict_tags.append(mapping_tags(predict_tagging_sequence))


print("Predicted tagging sequences (0-4): ", predict_tags[0:5])

# Print out the evaluations
accuracy = compute_accuracy(predict_tags, val_data_tags)
predicate_match_score = compute_predMatch(predict_tags, val_data_tags, val_data_sents, ignoreStopwords=True, ignoreCase=True)
argument_match_score = compute_argMatch(predict_tags, val_data_tags, val_data_sents, ignoreStopwords=True, ignoreCase=True)
print("The accuracy is: %f" % (accuracy))
print("The predicate matching score is %f" % (predicate_match_score))
print("The argument matching score is %f" % (argument_match_score))