"""main moduleof translation programm
"""
import sys
import os
import operator
import getopt
import time
import translator as tr
import re
import shutil
import nltk.stem
import numpy

# coding: utf8


def main():

    start_time = time.time()
    DIR = "./latters/translated"
    ordered_files = sorted(os.listdir(
        DIR), key=lambda x: int(re.search(r'\d+', x).group()))

    latters = [open(os.path.join(DIR, f)).read() for f in ordered_files]
    r = re.compile(r'\d')
    latters = [r.sub('', x) for x in latters]

    stoplist = []
    with open('stopwords.txt', 'r') as f:
        stoplist = f.read().split()

    proc_latters = [[word for word in latter.lower().split() if word not in stoplist]
                    for latter in latters]

    from gensim import corpora, models

    from collections import defaultdict
    frequency = defaultdict(int)
    for latter in proc_latters:
        for token in latter:
            frequency[token] += 1

    proc_latters = [[token for token in text if frequency[token] > 10 and frequency[token] < len(frequency) * 0.5]
                    for text in proc_latters]

    dictionary = corpora.Dictionary(proc_latters)
    dictionary.save('/tmp/latters.dict')

    corpus = [dictionary.doc2bow(text) for text in proc_latters]
    corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
    corpus = corpora.BleiCorpus('/tmp/corpus.lda-c')
    model = models.ldamodel.LdaModel(
        corpus,
        id2word=corpus.id2word)
    

    doc = corpus.docbyoffset(0)
    topics = model[doc]
    topics = [model[corpus[x]] for x in range(0, len(corpus))]

    print(topics)
    import matplotlib.pyplot as plt
    num_topics_used = [len(model[doc]) for doc in corpus]
    plt.hist(num_topics_used)
    plt.show()

if __name__ == '__main__':
    main()
