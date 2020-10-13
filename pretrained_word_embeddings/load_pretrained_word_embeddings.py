import bcolz as bcolz
import pickle as pickle
import numpy as np

"""
Load sentences from data file
"""
def select_row_data(filename):
    all_sents = open(filename, "r")
    all_data_sentences = (all_sents.read()).split("\n")
    return all_data_sentences


"""
Create the weight matrix for the BiLSTM-CRF model
Add words not in GloVe but in the training sentences into the weight matrix
Assign a random embedding vector to each of non-GloVe word
"""
def create_weight_matrix(filepath):
	words = []
	idx = 0
	word_to_ix = {}
	vectors = bcolz.carray(np.zeros(1), rootdir = f'../pretrained_word_embeddings/GloVe/6B.50.dat', mode = 'w')

	with open(f'../pretrained_word_embeddings/GloVe/glove.6B.50d.txt', 'rb') as f:
	    for l in f:
	        line = l.decode().split()
	        word = line[0]
	        words.append(word)
	        word_to_ix[word] = idx
	        idx += 1
	        vect = np.array(line[1:]).astype(np.float)
	        vectors.append(vect)

	vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir = f'../pretrained_word_embeddings/GloVe/6B.50.dat', mode = 'w')
	vectors.flush()
	pickle.dump(words, open(f'../pretrained_word_embeddings/GloVe/6B.50_words.pkl', 'wb'))
	pickle.dump(word_to_ix, open(f'../pretrained_word_embeddings/GloVe/6B.50_word_idx.pkl', 'wb'))

	vectors = bcolz.open(f'../pretrained_word_embeddings/GloVe/6B.50.dat')[:]
	words = pickle.load(open(f'../pretrained_word_embeddings/GloVe/6B.50_words.pkl', 'rb'))
	word_to_ix = pickle.load(open(f'../pretrained_word_embeddings/GloVe/6B.50_word_idx.pkl', 'rb'))

	glove = {w: vectors[word_to_ix[w]] for w in words}

	all_data_sentences = select_row_data(filepath)
	all_data = [all_data_sentences[i].split() for i in range(len(all_data_sentences))]
	for sentence in all_data:
	    for word in sentence:
	        if word not in words and len(word) > 0:
	            words.append(word)
	            word_to_ix[word] = idx
	            idx += 1

	pickle.dump(words, open(f'../pretrained_word_embeddings/GloVe/6B.50_words.pkl', 'wb'))
	pickle.dump(word_to_ix, open(f'../pretrained_word_embeddings/GloVe/6B.50_word_idx.pkl', 'wb'))
            

	matrix_len = len(words)
	weights_matrix = np.zeros((matrix_len, 50))
	words_found = 0

	print("Total words: ", len(words))

	for i, word in enumerate(words):
	    try:
	        weights_matrix[i] = glove[word]
	        words_found += 1
	    except KeyError:
	        weights_matrix[i] = np.random.normal(scale = 0.6, size = (50, ))

	pickle.dump(weights_matrix, open(f'../pretrained_word_embeddings/GloVe/6B.50_weights_matrix.pkl', 'wb'))

	print("Pretrained word vectors number: ", words_found)
	return weights_matrix

weights_matrix = create_weight_matrix("../data/train+val.sents.txt")