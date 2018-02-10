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
								host='127.0.0.1', database='ta-quran-0-4800')
	cursor = cnx.cursor()
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

# get semua data class 
def getAllClassList():
	className   = []
	query       = ("SELECT DISTINCT level_1 from ta_kelas")
	cursor.execute(query,)

	temp = []
	for level_1, in cursor:
		temp.append(level_1)
	return temp

def preProcessing(terjemahan):
	letters_only	= re.sub("[^a-zA-Z]"," ",terjemahan)
	words 			= letters_only.lower().split()
	stops 			= set(stopwords.words("english"))
	real_words 		= [w for w in words if not w in stops]
	return(" ".join(real_words))

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
	countWordArray  = []

	for tag, count in zip(vocab,dist):
		wordArray.append(tag)
		countWordArray.append(count)

	return wordArray, countWordArray

# store prior ke database
def storePrior(rangeawal, rangeakhir, level_1, pembagi):
	query		= ("SELECT COUNT(DISTINCT id_ayat) AS Count FROM ta_kelas WHERE (id_ayat > %s AND id_ayat <= %s) AND level_1 = %s")
	cursor.execute(query,(rangeawal, rangeakhir, level_1))
	prior 		= cursor.fetchone()[0] /pembagi

	prior_yes	= math.log(prior)
	prior_no	= math.log(1-prior)

	query       = ("INSERT INTO ta_prior VALUES (%s, %s, %s)")
	try:
		cursor.execute(query,(float(prior_yes), float(prior_no), level_1))
		cnx.commit()
	except:
		cnx.rollback()

# hitung likelihood
def storeLikelihood(dataWordAll, rangeawal, rangeakhir, level_1):
	for j in range(0, len(dataWordAll)):
		
		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE ((cleanword LIKE %s) OR (cleanword LIKE %s) OR (cleanword LIKE %s)) AND ((id_ayat > %s) AND (id_ayat <= %s)) AND (level_1 = %s)")
		cursor.execute(query,('% '+dataWordAll[j]+' %', '% '+dataWordAll[j]+'%', '%'+dataWordAll[j]+' %', rangeawal, rangeakhir, level_1))
		TT = cursor.fetchone()[0] + 1

		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE ((cleanword NOT LIKE %s) OR (cleanword NOT LIKE %s) OR (cleanword NOT LIKE %s)) AND ((id_ayat) > %s AND (id_ayat <= %s)) AND (level_1 = %s)")
		cursor.execute(query,('% '+dataWordAll[j]+' %', '% '+dataWordAll[j]+'%', '%'+dataWordAll[j]+' %', rangeawal, rangeakhir, level_1))
		TF = cursor.fetchone()[0] + 1

		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE ((cleanword LIKE %s) OR (cleanword LIKE %s) OR (cleanword LIKE %s)) AND ((id_ayat) > %s AND (id_ayat <= %s)) AND (level_1 <> %s)")
		cursor.execute(query,('% '+dataWordAll[j]+' %', '% '+dataWordAll[j]+'%', '%'+dataWordAll[j]+' %', rangeawal, rangeakhir, level_1))
		FT = cursor.fetchone()[0] + 1

		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE ((cleanword NOT LIKE %s) OR (cleanword NOT LIKE %s) OR (cleanword NOT LIKE %s)) AND ((id_ayat) > %s AND (id_ayat <= %s)) AND (level_1 <> %s)")
		cursor.execute(query,('% '+dataWordAll[j]+' %', '% '+dataWordAll[j]+'%', '%'+dataWordAll[j]+' %', rangeawal, rangeakhir, level_1))
		FF = cursor.fetchone()[0] + 1

		sumAllTrue	= TT + TF
		sumAllFalse = FT + FF
		
		query  = ("INSERT INTO ta_likelihood VALUES (%s, %s, %s, %s, %s, %s)")
		try:
			cursor.execute(query,(dataWordAll[j], level_1, float(math.log(TT/sumAllTrue)), float(math.log(TF/sumAllTrue)), float(math.log(FT/sumAllFalse)), float(math.log(FF/sumAllFalse))))
			cnx.commit()
		except:
			cnx.rollback()

# untuk get berapa mutual yang akan digunakan
def getMutual(threshold):
	query = ("SELECT parent, child, prob FROM ta_mutual ORDER BY prob DESC")
	cursor.execute(query)

	p = []
	c = []
	for parent, child, prob in cursor:
		if prob >= threshold:
			if (parent not in p) and (child not in c) and (parent not in c) and (child not in p):
				p.append(parent)
				c.append(child)
	return p, c

# hitung probability dengan 2 parent
def storeTwoParents(rangeawal, rangeakhir, arrP, arrCh, level_1):
	for j in range(0, len(arrP)):
		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE (level_1 = %s) AND (((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s))) AND (id_ayat > %s AND id_ayat <= %s)")
		cursor.execute(query,(level_1, '% '+arrP[j]+' %', '% '+arrCh[j]+' %', '% '+arrP[j]+' %', '%'+arrCh[j]+' %', '% '+arrP[j]+' %', '% '+arrCh[j]+'%', '%'+arrP[j]+' %', '% '+arrCh[j]+' %', '%'+arrP[j]+' %', '%'+arrCh[j]+' %', '%'+arrP[j]+' %', '% '+arrCh[j]+'%', '% '+arrP[j]+'%', '% '+arrCh[j]+' %', '% '+arrP[j]+'%', '%'+arrCh[j]+' %', '% '+arrP[j]+'%', '% '+arrCh[j]+'%', rangeawal, rangeakhir))
		TTT = cursor.fetchone()[0] + 1

		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE (level_1 = %s) AND (((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s))) AND (id_ayat > %s AND id_ayat <= %s)")
		cursor.execute(query,(level_1, '% '+arrP[j]+' %', '% '+arrCh[j]+' %', '% '+arrP[j]+' %', '%'+arrCh[j]+' %', '% '+arrP[j]+' %', '% '+arrCh[j]+'%', '%'+arrP[j]+' %', '% '+arrCh[j]+' %', '%'+arrP[j]+' %', '%'+arrCh[j]+' %', '%'+arrP[j]+' %', '% '+arrCh[j]+'%', '% '+arrP[j]+'%', '% '+arrCh[j]+' %', '% '+arrP[j]+'%', '%'+arrCh[j]+' %', '% '+arrP[j]+'%', '% '+arrCh[j]+'%', rangeawal, rangeakhir))
		TTF = cursor.fetchone()[0] + 1

		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE (level_1 = %s) AND (((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s))) AND (id_ayat > %s AND id_ayat <= %s)")
		cursor.execute(query,(level_1, '% '+arrP[j]+' %', '% '+arrCh[j]+' %', '% '+arrP[j]+' %', '%'+arrCh[j]+' %', '% '+arrP[j]+' %', '% '+arrCh[j]+'%', '%'+arrP[j]+' %', '% '+arrCh[j]+' %', '%'+arrP[j]+' %', '%'+arrCh[j]+' %', '%'+arrP[j]+' %', '% '+arrCh[j]+'%', '% '+arrP[j]+'%', '% '+arrCh[j]+' %', '% '+arrP[j]+'%', '%'+arrCh[j]+' %', '% '+arrP[j]+'%', '% '+arrCh[j]+'%', rangeawal, rangeakhir))
		TFT = cursor.fetchone()[0] + 1

		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE (level_1 = %s) AND (((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s))) AND (id_ayat > %s AND id_ayat <= %s)")
		cursor.execute(query,(level_1, '% '+arrP[j]+' %', '% '+arrCh[j]+' %', '% '+arrP[j]+' %', '%'+arrCh[j]+' %', '% '+arrP[j]+' %', '% '+arrCh[j]+'%', '%'+arrP[j]+' %', '% '+arrCh[j]+' %', '%'+arrP[j]+' %', '%'+arrCh[j]+' %', '%'+arrP[j]+' %', '% '+arrCh[j]+'%', '% '+arrP[j]+'%', '% '+arrCh[j]+' %', '% '+arrP[j]+'%', '%'+arrCh[j]+' %', '% '+arrP[j]+'%', '% '+arrCh[j]+'%', rangeawal, rangeakhir))
		TFF = cursor.fetchone()[0] + 1

		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE (((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword LIKE %s))) AND (id_ayat > %s AND id_ayat <= %s)")
		cursor.execute(query,('% '+arrP[j]+' %', '% '+arrCh[j]+' %', '% '+arrP[j]+' %', '%'+arrCh[j]+' %', '% '+arrP[j]+' %', '% '+arrCh[j]+'%', '%'+arrP[j]+' %', '% '+arrCh[j]+' %', '%'+arrP[j]+' %', '%'+arrCh[j]+' %', '%'+arrP[j]+' %', '% '+arrCh[j]+'%', '% '+arrP[j]+'%', '% '+arrCh[j]+' %', '% '+arrP[j]+'%', '%'+arrCh[j]+' %', '% '+arrP[j]+'%', '% '+arrCh[j]+'%', rangeawal, rangeakhir))
		FTT = cursor.fetchone()[0] - TTT + 2

		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE (((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword LIKE %s) AND (cleanword NOT LIKE %s))) AND (id_ayat > %s AND id_ayat <= %s)")
		cursor.execute(query,('% '+arrP[j]+' %', '% '+arrCh[j]+' %', '% '+arrP[j]+' %', '%'+arrCh[j]+' %', '% '+arrP[j]+' %', '% '+arrCh[j]+'%', '%'+arrP[j]+' %', '% '+arrCh[j]+' %', '%'+arrP[j]+' %', '%'+arrCh[j]+' %', '%'+arrP[j]+' %', '% '+arrCh[j]+'%', '% '+arrP[j]+'%', '% '+arrCh[j]+' %', '% '+arrP[j]+'%', '%'+arrCh[j]+' %', '% '+arrP[j]+'%', '% '+arrCh[j]+'%', rangeawal, rangeakhir))
		FTF = cursor.fetchone()[0] - TTF + 2

		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE (((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword LIKE %s))) AND (id_ayat > %s AND id_ayat <= %s)")
		cursor.execute(query,('% '+arrP[j]+' %', '% '+arrCh[j]+' %', '% '+arrP[j]+' %', '%'+arrCh[j]+' %', '% '+arrP[j]+' %', '% '+arrCh[j]+'%', '%'+arrP[j]+' %', '% '+arrCh[j]+' %', '%'+arrP[j]+' %', '%'+arrCh[j]+' %', '%'+arrP[j]+' %', '% '+arrCh[j]+'%', '% '+arrP[j]+'%', '% '+arrCh[j]+' %', '% '+arrP[j]+'%', '%'+arrCh[j]+' %', '% '+arrP[j]+'%', '% '+arrCh[j]+'%', rangeawal, rangeakhir))
		FFT = cursor.fetchone()[0] - TFT + 2

		query	= ("SELECT COUNT(*) FROM ta_kelas WHERE (((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s)) OR ((cleanword NOT LIKE %s) AND (cleanword NOT LIKE %s))) AND (id_ayat > %s AND id_ayat <= %s)")
		cursor.execute(query,('% '+arrP[j]+' %', '% '+arrCh[j]+' %', '% '+arrP[j]+' %', '%'+arrCh[j]+' %', '% '+arrP[j]+' %', '% '+arrCh[j]+'%', '%'+arrP[j]+' %', '% '+arrCh[j]+' %', '%'+arrP[j]+' %', '%'+arrCh[j]+' %', '%'+arrP[j]+' %', '% '+arrCh[j]+'%', '% '+arrP[j]+'%', '% '+arrCh[j]+' %', '% '+arrP[j]+'%', '%'+arrCh[j]+' %', '% '+arrP[j]+'%', '% '+arrCh[j]+'%', rangeawal, rangeakhir))
		FFF = cursor.fetchone()[0] - TFF + 2

		sumAllTT	= TTT + TTF
		sumAllTF 	= TFT + TFF
		sumAllFT	= FTT + FTF
		sumAllFF 	= FFT + FFF

		query       = ("INSERT INTO ta_cpt VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
		try:
			cursor.execute(query,(arrP[j], arrCh[j], level_1, float(math.log(TTT/sumAllTT)), float(math.log(TTF/sumAllTT)), float(math.log(TFT/sumAllTF)), float(math.log(TFF/sumAllTF)), float(math.log(FTT/sumAllFT)), float(math.log(FTF/sumAllFT)), float(math.log(FFT/sumAllFF)), float(math.log(FFF/sumAllFF))))
			cnx.commit()
		except:
			cnx.rollback()

# MAIN

rangeawal	= 0
rangeakhir	= 4800

arrKelas					= getAllClassList()
arrP, arrCh 				= getMutual(3)
print (len(arrCh))
for i in range(0, len(arrKelas)):
	print (arrKelas[i])

	level_1 = arrKelas[i]
	storePrior(rangeawal, rangeakhir, level_1, 4800)
	storeLikelihood(dataWordAll, rangeawal, rangeakhir, level_1)
	storeTwoParents(rangeawal, rangeakhir, arrP, arrCh, level_1)

print ("Bismillah mudah-mudahan ada keajaiban dari Allah, Aamiin")