import numpy as np
import re
import nltk
import sys
import skTest
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
# import numpy as np
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


documents = []

""" df = pd.read_csv(".\models.csv", header = 0);
original_headers = list(df.columns.values);
 """
df = pd.read_csv(".\models.csv");


# movie_data = load_files( ".\models.csv" , categories=['summary', 'ok', 'virus'])
X, y = df['sentence'], df['tag']


stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

tfidfconverter = TfidfVectorizer(
    max_features=1500, min_df=0, max_df=1, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()
xNew = tfidfconverter.fit_transform(['D:\GameApsdk\Isdnto.tasdhe.Dead.2.v0.8.1_androidbazaar.com.apk: OK']).toarray()
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X, y)

yNew = classifier.predict(xNew)
