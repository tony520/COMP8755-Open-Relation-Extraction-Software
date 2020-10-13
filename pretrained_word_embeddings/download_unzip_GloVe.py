"""
In 2020, we used dload - a new Python library for downloading and extracting
pre-trained word embeddings GloVe
"""
import dload

def download_unzip_pretrained_word_embeddings(url, save_path):
	dload.save_unzip(url, save_path, True)
	print("Finishing download and unzip for GloVe!")

download_unzip_pretrained_word_embeddings("http://nlp.stanford.edu/data/glove.6B.zip", "./GloVe")