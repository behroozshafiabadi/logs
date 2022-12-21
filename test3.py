import multiprocessing
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import seaborn as sns
import re
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import gensim
from sklearn.model_selection import train_test_split
from skTest import utils
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
cores = multiprocessing.cpu_count()


df = pd.read_csv('models.csv')
df = df[['sentence', 'tag']]
df = df[pd.notnull(df['sentence'])]
df.rename(columns={'sentence': 'sentence'}, inplace=True)
df.head(10)

train, test = train_test_split(df, test_size=0.01, random_state=42)
sample_test = test[:1000]
sample_train = train[:2000]


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


train_tagged = sample_train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['sentence']), tags=[r.tag]), axis=1)
test_tagged = sample_test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['sentence']), tags=[r.tag]), axis=1)

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5,
                     hs=0, min_count=2, sample=0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

model.save("d2v.model")

print("Model Saved")

model= Doc2Vec.load("d2v.model")

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(
        *[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors


y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(
    f1_score(y_test, y_pred, average='weighted')))

train_tagged.values[30]
