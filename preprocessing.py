import pandas as pd

def list_to_string(in_list):
    strg = ''
    strg = ' '.join([str(elem) for elem in in_list])
    return strg

def convert_conll_to_txt(IN_FILE, OUT_FILE_SENTS, OUT_FILE_TAGS):
	sents_file = open(OUT_FILE_SENTS, 'w')
	tags_file = open(OUT_FILE_TAGS, 'w')

	data = pd.read_csv(IN_FILE, sep='\t', usecols=['word', 'label'])
	df = pd.DataFrame(data)

	words = []
	labels = []
	for index, row in df.iterrows():
	    print(row['word'])
	    if row['word'] != '.' and row['word'] != '\n':
	        words.append(str(row['word']).lower())
	        labels.append(row['label'])
	    else:
	        #words.append(row['word'])      # do not include final dot
	        #labels.append(row['label'])    # do not include final dot
	        instance = (words, labels)
	        #print('instance', instance)
	        # training_data.append(instance)
	        sents_file.write(list_to_string(words) + '\n')
	        tags_file.write(list_to_string(labels) + '\n')
	        words = []
	        labels = []

	sents_file.close()
	tags_file.close()
	print("Done. The training data (.txt) have been saved in data folder")


convert_conll_to_txt("data/train.conll", "data/train.sents.txt", "data/train.tags.txt")
convert_conll_to_txt("data/val.conll", "data/val.sents.txt", "data/val.tags.txt")
convert_conll_to_txt("data/train+val.conll", "data/train+val.sents.txt", "data/train+val.tags.txt")