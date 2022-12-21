# Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize  # Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

similarity_sententenses = [
    "number of viruses find: the count of virus, known viruses, scan summary, engine version, scanned directories, scanned files, infected files, data scanned, data read, ----------- SCAN SUMMARY ----------- Known viruses: Engine version: Scanned directories:  Scanned files:   Infected files:  Data scanned:  Data read: MB (ratio 2.07:1)",
    "winpath: ok",
    "unixpath: ok",
    "winpath: found",
    "unixpath: found"
]

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[
                              str(i)]) for i, _d in enumerate(similarity_sententenses)]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

model.train(tagged_data, total_examples=model.corpus_count,
            epochs=model.epochs)

model.save("d2v.model")

print("Model Saved")

model= Doc2Vec.load("d2v.model")

test_doc = word_tokenize("WINPATH OK".lower())

top = model.dv.most_similar(positive=[model.infer_vector(test_doc)],topn=3)

print("top :", top)
