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


def find_topics_count(topics):
    proc_topics = []

    for topic in topics:
        x = re.sub(r'\d+\.\d+\*', '', topic[1])
        x = re.sub(r'[\"\+]', '', x)
        proc_topics.append(x.split())

    k = 1
    for_del = []

    for x in proc_topics:
        count = 0
        for y in proc_topics[k:]:
            for z in x:
                if z in y:
                    count += 1
            perc = count /  len(x)
            if perc >= 0.4:
                for_del.append(k-1)
                break
            count = 0
        k += 1
    
    for x in for_del:
        proc_topics[x] = None

    return len(for_del)


def main():

    start_time = time.time()
    DIR = "./letters/translated"
    ordered_files = sorted(os.listdir(
        DIR), key=lambda x: int(re.search(r'\d+', x).group()))

    letters = [open(os.path.join(DIR, f)).read() for f in ordered_files]
    r = re.compile(r'\d')
    letters = [r.sub('', x) for x in letters]

    stoplist = []
    with open('stopwords.txt', 'r') as f:
        stoplist = f.read().split(',')

    proc_letters = [tr.mail_process(text) for text in letters]
    proc_letters = [[word for word in latter.lower().split() if word not in stoplist]
                    for latter in proc_letters]

    from gensim import corpora, models

    from collections import defaultdict
    frequency = defaultdict(int)
    for latter in proc_letters:
        for token in latter:
            frequency[token] += 1

    proc_letters = [[token for token in text if frequency[token] > 10 and frequency[token] < len(frequency) * 0.5]
                    for text in proc_letters]

    dictionary = corpora.Dictionary(proc_letters)
    dictionary.save('./tmp/letters.dict')
    id2word = dictionary

    corpus = [id2word.doc2bow(text) for text in proc_letters]
    corpora.MmCorpus.serialize('./tmp/corpus.mm', corpus)
    corpus = corpora.MmCorpus('./tmp/corpus.mm')

    model = models.hdpmodel.HdpModel(
        corpus=corpus,
        id2word=id2word)

    all_topics = model.show_topics(-1, num_words=10)

    all_topics = sorted(all_topics, key=lambda x: x[0])

    find_topics_count(all_topics)

    # print(all_topics)

    topics = [model[doc] for doc in corpus]
    # print(topics)

    tfidf = models.TfidfModel(corpus)

    with open('latter-topics.txt', 'w') as f:
        for x in zip(ordered_files, range(0, len(corpus))):
            f.write('{}: {}\n'.format(x[0], model[corpus[x[1]]]))

    with open('topics.txt', 'w') as f:
        for top in all_topics:
            f.write('{}\n'.format(top))

    # topics = model[doc]
    # topics = [model[corpus[x]] for x in range(0, len(corpus))]

    # for x in topics:
    #     print(x)
    import matplotlib.pyplot as plt
    num_topics_used = [len(model[doc]) for doc in corpus]
    plt.hist(num_topics_used)
    plt.show()


if __name__ == '__main__':
    main()
