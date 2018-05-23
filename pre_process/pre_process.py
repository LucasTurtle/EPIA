import numpy as np
import pandas as pd
import os
import re

from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

import matplotlib.pylab as plt
import time



def loadScrapmaker(sourcepath):
	scrapmaker = []
	for filename in os.listdir(sourcepath):
		with open(os.path.join(sourcepath, filename), 'r') as ins:
			for line in ins:
				scrapmaker.append(line)
	scrapmaker = [x.strip() for x in scrapmaker] 
	scrapmaker = [x.casefold() for x in scrapmaker] 
	return scrapmaker

def removeWords(listOfTokens, listOfWords):
	return [token for token in listOfTokens if token not in listOfWords]

def applyStemming(listOfTokens, stemmer):
	return [stemmer.stem(token) for token in listOfTokens]

def twoLetters(listOfTokens):
	twoLetterWord = []
	for token in listOfTokens:
		if len(token) <= 2:
			twoLetterWord.append(token)
	return twoLetterWord

def processCorpus(corpus):
	stop_words = list(get_stop_words('en'))         # 900 stopwords
	nltk_words = list(stopwords.words('english'))   # 150 stopwords
	scrapmaker = loadScrapmaker(os.path.join('.', 'scrapmaker/'))
	param_stemmer = SnowballStemmer('english')
	for document in corpus:
		index = corpus.index(document)
		corpus[index] = corpus[index].rstrip('\n')
		corpus[index] = corpus[index].casefold()
		corpus[index] = re.sub("\S*@\S*\s?"," ", corpus[index])
		corpus[index] = re.sub('\W_',' ', corpus[index])        # remove caracteres especiais e deixa somente palavras
		corpus[index] = re.sub("\S*\d\S*"," ", corpus[index])   # remove numeros e palavras juntas de numeros IE h4ck3r
		corpus[index] = re.sub("\S*@\S*\s?"," ", corpus[index]) # remove endereços de email
		corpus[index] = corpus[index].replace("_", " ")         # remove underscore e substitui por whitespace
		corpus[index] = corpus[index].replace("-", " ")         # remove sinal de menos e substitui por whitespace
		listOfTokens = word_tokenize(corpus[index])
		listOfTokens = removeWords(listOfTokens, stop_words)
		listOfTokens = removeWords(listOfTokens, nltk_words)
		listOfTokens = removeWords(listOfTokens, scrapmaker)
		twoLetterWord = twoLetters(listOfTokens)
		listOfTokens = removeWords(listOfTokens, twoLetterWord)
		listOfTokens = applyStemming(listOfTokens, param_stemmer)

		corpus[index]   = " ".join(listOfTokens)
	return corpus

# especifica qual topico tal documento faz parte
def setTargets(dataset, path):
	elements = dataset.filenames.tolist()
	elements = [element.split('in/')[-1] for element in elements]
	elements = [element.split('/')[0] for element in elements]
	df = pd.DataFrame({'Name':elements, 'Target': dataset.target})
	save_to = path + '1_target.tsv'
	df.to_csv(save_to, sep='\t')

def process(data, path):
	save_to = path + "binary.tsv"

	# processamento binário
	vecBin = CountVectorizer(binary=True)
	binary = vecBin.fit_transform(data)
	dfBinary = pd.DataFrame(binary.toarray(), columns=vecBin.get_feature_names(), dtype='float')
	dfBinary.to_csv(save_to, sep='\t')

	# TF
	save_to = path + "tf.tsv"
	vec = CountVectorizer(stop_words='english')
	tf = vec.fit_transform(data)
	dfTF = pd.DataFrame(tf.toarray(), columns=vec.get_feature_names(), dtype='float')
	dfTF.to_csv(save_to, sep='\t')

	#ocorrencia de palavras
	save_to = path + "2_word_count.tsv"
	b = dfTF.max(axis=0)
	b.sort_values(ascending=False).to_csv(save_to, sep='\t')

	#TF-IDF
	save_to = path + "tfidf.tsv"
	transformer = TfidfTransformer(smooth_idf=False)
	tfidf = transformer.fit_transform(tf)
	dfIDF = pd.DataFrame(tfidf.toarray(), columns=vec.get_feature_names(), dtype='float')
	dfIDF.to_csv(save_to, sep='\t')


def main():
	start = time.time()

	#categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'sci.electronics', 'rec.sport.baseball', 'rec.sport.hockey', 'talk.politics.guns', 'talk.politics.mideast', 'talk.religion.misc', 'soc.religion.christian']
	categories = ['sci.space']
	twenty_dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),categories=categories, shuffle=True, random_state=42)
	print(twenty_dataset.filenames.shape)
	corpus = processCorpus(twenty_dataset.data)

	if (len(categories) == 1):
		path = 'processedSmall/'
	else:
		path = 'processedBig/'

	setTargets(twenty_dataset, path)
	process(corpus, path)

	end = time.time()
	temp = end-start

	hours = temp//3600
	temp = temp - 3600*hours
	minutes = temp//60
	seconds = temp - 60*minutes
	print('%d:%d:%d' %(hours,minutes,seconds))

	save_to = path + "0_elapsed_time.txt"
	text_file = open(save_to, "w")
	text_file.write('%d:%d:%d' %(hours,minutes,seconds))
	text_file.close()

if __name__ == '__main__':
	main()