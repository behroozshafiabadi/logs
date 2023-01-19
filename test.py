import re
import pandas as pd
import os
import copy
import alignment
import hashlib
from datetime import datetime
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# import spacy
# from spacy.matcher import Matcher

# nlp = spacy.load("en_core_web_sm")
# matcher = Matcher(nlp.vocab)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


rex = {
    "IP": r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',  # ip regex
    "WINPATH": r'[^-\s]([^-\w*]\\\\[\w*\\{\}:`!@#_.\-]+)[^-\s]',
    "UNIXPATH": r'[^-\s]([^-\w*]\/[\w\/{\}:`!@#_.\-]+)[^-\s]',
    "NUMBER": r'[-+]?\d*\.*\d+'  # numbers and floating points
}

levels = 2
level_clusters = {}
df_log = None
path = ""
k = 1
k1 = 1
k2 = 1
max_dist = 0.001
alpha = 100
savePath = os.getcwd()
logname = "logs.csv"
tfidfconverter = TfidfVectorizer(
    max_features=1500, min_df=0, max_df=1, stop_words=stopwords.words('english'))
model = None


class partition():
    def __init__(self, idx, log="", lev=-1, group=""):
        self.logs_idx = [idx]
        self.patterns = [log]
        self.level = lev
        self.group = [group]

# r'\\(\\[\w()\[\]\{\}:`!@#_\-]+)*\\',


def pre_process(array_sentences):
    documents = []
    stemmer = WordNetLemmatizer()

    for sen in range(0, len(array_sentences)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(array_sentences[sen]))

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

    return documents

# TODO : add file_path argument for dynamicly reading files


def classification():

    cf_models = pd.read_csv(".\models.csv")
    X, y = cf_models['sentence'], cf_models['tag']

    X = pre_process(X)
    # tfidfconverter = TfidfVectorizer(
    #     max_features=1500, min_df=0, max_df=1, stop_words=stopwords.words('english'))
    X = tfidfconverter.fit_transform(X).toarray()

    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X, y)
    return classifier


def predict(sentneces, model):
    xNew = pre_process([sentneces])
    xNew = tfidfconverter.transform(xNew).toarray()
    yNew = model.predict(xNew)
    return yNew


def parse(logname):
    print('Parsing file: ' + os.path.join(path, logname))
    logname = logname
    starttime = datetime.now()
    global df_log
    global model
    # TODO: must load dynamicly
    df_log = load_data()
    model = classification()
    for lev in range(levels):
        if lev == 0:
            # Clustering
            level_clusters[0] = get_clusters(
                df_log['Content_'], lev)
        else:
            # Clustering
            patterns = [c.patterns[0] for c in level_clusters[lev-1]]
            global max_dist
            max_dist *= alpha
            clusters = get_clusters(
                patterns, lev, level_clusters[lev-1])

            # Generate patterns
            for cluster in clusters:
                cluster.patterns = [sequential_merge(cluster.patterns)]
            level_clusters[lev] = clusters
    dump()
    print('Parsing done. [Time taken: {!s}]'.format(
        datetime.now() - starttime))


def sequential_merge(logs):
    log_merged = logs[0]
    for log in logs[1:]:
        log_merged = pair_merge(log_merged, log)
    return log_merged


def pair_merge(loga, logb):
    loga, logb = alignment.water(loga.split(), logb.split())
    logn = []
    for idx, value in enumerate(loga):
        logn.append('<*>' if value != logb[idx] else value)
    return " ".join(logn)


def get_clusters(logs, lev, old_clusters=None):

    clusters = []
    old_clusters = copy.deepcopy(old_clusters)
    for logidx, log in enumerate(logs):
        match = False
        for cluster in clusters:
            dis = msgDist(log, cluster.patterns[0]) if lev == 0 else patternDist(
                log, cluster.patterns[0])
            if isinstance(dis, (int, float)) and dis < max_dist:
                if lev == 0:
                    cluster.logs_idx.append(logidx)
                else:
                    cluster.logs_idx.extend(old_clusters[logidx].logs_idx)
                    cluster.patterns.append(old_clusters[logidx].patterns[0])
                    if cluster.group[0] != old_clusters[logidx].group[0]:
                        cluster.group.append(old_clusters[logidx].group[0])
                match = True

        if not match:
            if lev == 0:
                # generate new cluster
                cluster_array = predict(log, model)
                clusters.append(partition(logidx, log, lev, cluster_array[0]))
            else:
                old_clusters[logidx].level = lev
                clusters.append(old_clusters[logidx])  # keep old cluster

    return clusters


def msgDist(seqP, seqQ):
    dis = 1
    seqP = seqP.split()
    seqQ = seqQ.split()
    maxlen = max(len(seqP), len(seqQ))
    minlen = min(len(seqP), len(seqQ))
    for i in range(minlen):
        dis -= (k if seqP[i] == seqQ[i] else 0 * 1.0) / maxlen
    return dis


def patternDist(seqP, seqQ):
    dis = 1
    seqP = seqP.split()
    seqQ = seqQ.split()
    maxlen = max(len(seqP), len(seqQ))
    minlen = min(len(seqP), len(seqQ))
    for i in range(minlen):
        if seqP[i] == seqQ[i]:
            if seqP[i] == "<*>":
                dis -= k2 * 1.0 / maxlen
            else:
                dis -= k1 * 1.0 / maxlen
    return dis


def dump():

    global df_log
    if not os.path.isdir(savePath):
        os.makedirs(savePath)
    templates = [0] * df_log.shape[0]
    ids = [0] * df_log.shape[0]
    groups = [0] * df_log.shape[0]
    templates_occ = defaultdict(int)

    for cluster in level_clusters[levels-1]:
        EventTemplate = cluster.patterns[0]
        EventId = hashlib.md5(
            ' '.join(EventTemplate).encode('utf-8')).hexdigest()[0:8]
        Occurences = len(cluster.logs_idx)  
        templates_occ[EventTemplate] += Occurences

        for idx in cluster.logs_idx:
            ids[idx] = EventId
            templates[idx] = EventTemplate
            groups[idx] = cluster.group[0]

    df_log['EventId'] = ids
    df_log['EventTemplate'] = templates

    occ_dict = dict(df_log['EventTemplate'].value_counts())
    df_event = pd.DataFrame()
    df_event['EventTemplate'] = df_log['EventTemplate'].unique()
    df_event['Occurrences'] = df_log['EventTemplate'].map(occ_dict)
    df_event['EventId'] = df_log['EventTemplate'].map(
        lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
    df_log.drop("Content_", inplace=True, axis=1)
    df_log.to_csv(os.path.join(savePath, logname +
                  '_structured.csv'), index=False)
    df_event.to_csv(os.path.join(savePath, logname + '_templates.csv'),
                    index=False, columns=["EventId", "EventTemplate", "Occurrences"])


def load_data():
    def preprocess(line):
        for key, currentRex in rex.items():
            line = re.sub(currentRex, key, line)
            line = line.strip('\"')
            line = line.replace("\\r\\n", "")
            line = line.strip()
        return line
    logformat = "<Time> <IP> <Content>"
    # logformat = "HELLO MAMAD by HELLO2 sasdasd"
    path = os.getcwd()
    headers, regex = generate_logformat_regex(logformat)
    df_log = log_to_dataframe(os.path.join(
        path, logname), regex, headers, logformat)
    df_log['Content_'] = df_log['Content'].map(preprocess)
    return df_log


def log_to_dataframe(log_file, regex, headers, logformat):
    ''' Function to transform log file to dataframe '''
    log_messages = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in fin:
            line.rstrip()
            try:
                match = re.search(regex, line.strip())
                """for header in headers:
                    message = match.group(header)
                    log_messages.append(message) """
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
        # linecount += 1
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf


def generate_logformat_regex(logformat):
    ''' 
    Function to generate regular expression to split log messages
    '''
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


# match = re.search('(?P<ad>.*) (?P<phone>.*)', 'John 123456')
# print("---->", match.group('ad'))

print(parse("test"))
