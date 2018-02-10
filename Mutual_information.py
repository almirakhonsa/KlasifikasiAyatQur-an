import pandas as pd
import re
import sys
import math
import numpy as np
from numpy import array
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import mysql.connector
from mysql.connector import errorcode
import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')
np.set_printoptions(threshold=sys.maxsize)

# bangun koneksi ke database
try:
	cnx = mysql.connector.connect(user='root', password='root',
		unix_socket = '/Applications/MAMP/tmp/mysql/mysql.sock',
		database='ta-quran-0-4236')
	cursor = cnx.cursor()
	cursor2 = cnx.cursor()
except mysql.connector.Error as err:
	if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
		print("Something is wrong with your user name or password")
	elif err.errno == errorcode.ER_BAD_DB_ERROR:
		print("Database does not exist")
	else:
		print(err)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

def preProcessing(terjemahan):
	letters_only	= re.sub("[^a-zA-Z]"," ",terjemahan)
	words 			= letters_only.lower().split()
	stops 			= set(stopwords.words("english"))
	real_words 		= [w for w in words if not w in stops]
	return(" ".join(real_words))

def getFeatures(data_train):
	vectorizer 		= StemmedCountVectorizer( 
											  # min_df		= 1,
											  analyzer		= "word",
											  tokenizer 	= None,
											  preprocessor 	= None,
											  stop_words	= None,
											  max_features	= 15000
												)

	dataFeatures 	= vectorizer.fit_transform(data_train)
	dataFeatures	= dataFeatures.toarray()
	vocab 			= vectorizer.get_feature_names()
	dist 			= np.sum(dataFeatures, axis=0)

	dataWord        = []
	dataCount       = []
	for word, count in zip(vocab,dist):
		dataWord.append(word)
		dataCount.append(count)

	return dataWord
#end of fungsi getFeatures

def makeDataSet(rangeawal, rangeakhir):
	query 	= ("SELECT id, teksayat FROM ta_ayat WHERE id > %s AND id <= %s")
	cursor.execute(query,(rangeawal,rangeakhir))

	id_data_training 	= []
	data_training		= []
	rows = cursor.fetchall()

	for row in rows:
		data_training.append(row[1])

	sz_dtTraining 	= len(data_training)
	clear_data 		= []

	for i in range(0,sz_dtTraining):
		clear_data.append(preProcessing(data_training[i]))
	
	return clear_data

def  mutualInformation(data_train, dataWord):
	print ("============= MUTUAL INFORMATION =============", "\n")
	N = len(data_train)
	
	for i in range(0, len(dataWord)):
		valMi = []
		mi = []
		for j in range(i, len(dataWord)):
			xy=1; x=1; y=1
			if dataWord[i] != dataWord [j]:
				for k in range(0, len(data_train)):
					if (' '+dataWord[i]+' ' in data_train[k])  and (' '+dataWord[j]+' ' in data_train[k]):
						xy = xy + 1
					elif (' '+dataWord[i]+' ' in data_train[k]) and (' '+dataWord[j]+' ' not in data_train[k]):
						x = x + 1
					elif (' '+dataWord[j]+' ' in data_train[k]) and (' '+dataWord[i]+' ' not in data_train[k]):
						y = y + 1
				val = math.log10((xy/N)/((x/N)*(y/N)))
				mi.append([val, dataWord[i], dataWord[j]])
		if mi != []:	
			x = max(mi)
			print ("Data ke - ",i, x)
			query = ("INSERT INTO ta_mutual VALUES (%s, %s, %s)")
			try:
				cursor.execute(query,(x[1], x[2], float(x[0])))
				cnx.commit()
			except:
				cnx.rollback()

data_train = makeDataSet(0, 6236)
dataWord = getFeatures(data_train)
x = mutualInformation(data_train, dataWord)