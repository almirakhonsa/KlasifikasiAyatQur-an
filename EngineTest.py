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

try:
	cnx = mysql.connector.connect(user='root', password='',
								 host='127.0.0.1',
								 database='ta-quran-0-4800')
	cursor = cnx.cursor(buffered=True)
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

#fungsi getFeatures
def getFeatures(data):
	vectorizer 		= StemmedCountVectorizer( 
											  # min_df		= 1,
											  analyzer		= "word",
											  tokenizer 	= None,
											  preprocessor 	= None,
											  stop_words	= None,
											  max_features	= 15000
												)

	dataFeatures 	= vectorizer.fit_transform(data)
	dataFeatures	= dataFeatures.toarray()
	vocab 			= vectorizer.get_feature_names()
	dist 			= np.sum(dataFeatures, axis=0)

	wordArray 		= []
	dataCount 		= []
	
	for tag, count in zip(vocab,dist):
		wordArray.append(tag)
		dataCount.append(count)
	return dataCount, wordArray

# get semua data class 
def getAllClassList():
	className   = []
	query       = ("SELECT DISTINCT level_1 from ta_kelas")
	cursor.execute(query,)

	temp = []
	for level_1, in cursor:
		temp.append(level_1)
	return temp

# untuk get berapa mutual yang akan digunakan
def getMutual(thresh):
	query = ("SELECT parent, child, prob FROM ta_mutual ORDER BY prob DESC")
	cursor.execute(query)

	p = []
	c = []
	for parent, child, prob in cursor:
		if (prob >= thresh):
			if ((parent not in p) and (child not in c)) and ((parent not in c) and (child not in p)):
				p.append(parent)
				c.append(child)
	return p, c

def preProcessing(terjemahan):
	letters_only	= re.sub("[^a-zA-Z]"," ",terjemahan)
	words 			= letters_only.lower().split()
	stops 			= set(stopwords.words("english"))
	real_words 		= [w for w in words if not w in stops]
	return(" ".join(real_words))

# ambil data test
def makeDataSet(rangeawal, rangeakhir):
	query 	= ("SELECT id, teksayat FROM ta_ayat WHERE id > %s AND id <= %s")
	cursor.execute(query,(rangeawal,rangeakhir))

	id_data_set			= []
	data_training		= []
	rows = cursor.fetchall()

	for row in rows:
		id_data_set.append(row[0])
		data_training.append(row[1])

	sz_dtTraining 	= len(data_training)
	clear_data 		= []

	for i in range(0,sz_dtTraining):
		clear_data.append(preProcessing(data_training[i]))
		
	return id_data_set, clear_data

# misal input ayat "beneficent merciful"
def posterior(arrCh, id_data_set, data_test, level_1):
	for i in range(0, len(data_test)):
		if data_test[i] != '':
			query	= ("SELECT SUM(TF), SUM(FF) FROM ta_likelihood WHERE level_1 = %s")
			cursor.execute(query,(level_1,))
			prob 	= cursor.fetchone()

			sumAllTrue = prob[0]
			sumAllFalse = prob[1]

			count, splitTrain 		= getFeatures([data_test[i]])

			for j in range(0, len(splitTrain)):
				query	= ("SELECT TT, TF, FT, FF FROM ta_likelihood WHERE word = %s AND level_1 = %s")
				cursor.execute(query,(splitTrain[j], level_1))
				cpt	= cursor.fetchone()

				TTWord	= cpt[0] * count[j]
				TFWord	= cpt[1] * count[j]
				FTWord	= cpt[2] * count[j]
				FFWord	= cpt[3] * count[j]

				if splitTrain[j] in arrCh:
					sumAllTrue	= (sumAllTrue - TFWord)
					sumAllFalse	= (sumAllFalse - FFWord)

				elif splitTrain[j] not in arrCh:
					sumAllTrue	= (sumAllTrue - TFWord) + TTWord
					sumAllFalse	= (sumAllFalse - FFWord) + FTWord

			for k in range(0, len(arrCh)):
				query	= ("SELECT parent, child, TTT, TTF, TFT, TFF, FTT, FTF, FFT, FFF FROM ta_cpt_split WHERE child = %s and level_1 = %s")
				cursor.execute(query,(arrCh[k], level_1))
				prob 	= cursor.fetchone()
						
				TTT		= prob[2] * count[j]
				TTF 	= prob[3] * count[j]
				TFT 	= prob[4] * count[j]
				TFF 	= prob[5] * count[j]
				FTT 	= prob[6] * count[j]
				FTF 	= prob[7] * count[j]
				FFT 	= prob[8] * count[j]
				FFF		= prob[9] * count[j]
					
				if (prob[0] in splitTrain) and (prob[1] in splitTrain):
					sumAllTrue	= sumAllTrue + TTT
					sumAllFalse	= sumAllFalse + FTT
				elif (prob[0] in splitTrain) and (prob[1] not in splitTrain):
					sumAllTrue	= sumAllTrue + TTF
					sumAllFalse	= sumAllFalse + FTF
				elif (prob[0] not in splitTrain) and (prob[1] in splitTrain):
					sumAllTrue	= sumAllTrue + TFT
					sumAllFalse	= sumAllFalse + FFT
				elif (prob[0] not in splitTrain) and (prob[1] not in splitTrain):
					sumAllTrue	= sumAllTrue + TFF
					sumAllFalse	= sumAllFalse + FFF

			query	= ("SELECT prior_yes, prior_no FROM ta_prior WHERE level_1 = %s")
			cursor.execute(query,(level_1,))
			prior 	= cursor.fetchone()

			posterior_yes 	= sumAllTrue + prior[0]
			posterior_no	= sumAllFalse + prior[1]
			
			if posterior_yes > posterior_no:
				query       = ("INSERT INTO ta_newoutput_split VALUES (%s, %s)")
				try:
					cursor.execute(query,(id_data_set[i], level_1))
					cnx.commit()
				except:
					cnx.rollback()
 
# MAIN
rangeawal	= 4800
rangeakhir	= 6236

arrKelas					= getAllClassList()
id_data_set, data_test		= makeDataSet(rangeawal, rangeakhir)
arrCh, arrP					= getMutual(4)

j = 0
for i in range(0, len(arrKelas)):
	j = j+1
	print (arrKelas[i])

	level_1 = arrKelas[i]
	posterior(arrCh, id_data_set, data_test, level_1)