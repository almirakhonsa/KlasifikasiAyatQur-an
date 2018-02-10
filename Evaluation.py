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
np.set_printoptions(threshold=sys.maxsize)

try:
  cnx = mysql.connector.connect(user='root', password='',
                 host='127.0.0.1',
                 database='ta-quran-1200-6236')
  cursor	= cnx.cursor();

except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)

def getTargetList(id_ayat):
  query = ("SELECT level_1 FROM ta_kelas WHERE id_ayat = %s")
  cursor.execute(query,(id_ayat,))
  temp1 = []
  for level_1, in cursor:
    if level_1 not in temp1:
      temp1.append(level_1)
  return temp1

def getOutputList(id_ayat):
  query = ("SELECT level_1 FROM ta_newoutput_split WHERE id_ayat = %s")
  cursor.execute(query,(id_ayat,))

  temp1 = []
  for level_1, in cursor:
    if level_1 not in temp1:
      temp1.append(level_1)
  return temp1

#MAIN
rangeawal = 0
rangeakhir = 1200

ctbenar = 0
ctsalah = 0
for i in range(rangeawal, rangeakhir):
  targetList  = getTargetList(i)
  outputList  = getOutputList(i)
  print (i)
  print (targetList)
  print (outputList)
  for j in range(0,len(outputList)):
    if(outputList[j] in targetList):
      ctbenar = ctbenar + 1
    else:
      ctsalah = ctsalah + 1

  for k in range(0,len(targetList)):
    if(targetList[k] in outputList):
      ctbenar = ctbenar + 1
    else:
      ctsalah = ctsalah + 1

print (ctbenar , ctsalah)
hammingLoss   = float(float(ctsalah) / float(15 * 1200)) * 100

print (hammingLoss)

# END OF MAIN