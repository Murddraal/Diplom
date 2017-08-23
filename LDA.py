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
from gensim import corpora, models
from collections import defaultdict
import matplotlib.pyplot as plt
import datetime
import codecs

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

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def processing_texts(letters):
    stoplist = []
    with open('stopwords.txt', 'r') as f:
        stoplist = f.read().split(',')

    try:
        proc_letters = [tr.mail_process(text) for text in letters]
    except:
        proc_letters = letters

    english_stemmer = nltk.stem.SnowballStemmer('english')
        
    proc_letters = [[english_stemmer.stem(word) for word in latter.lower().split() if word not in stoplist and is_int(word) == False]
                    for latter in proc_letters]
    
    frequency = defaultdict(int)
    for latter in proc_letters:
        for token in latter:
            frequency[token] += 1

    # proc_letters = [[token for token in text if frequency[token] > 3 and frequency[token] < len(frequency) * 0.7]
    #                 for text in proc_letters]
    
    return proc_letters

def read_letters():
    DIR = "./letters/translated"
    ordered_files = sorted(os.listdir(
        DIR), key=lambda x: int(re.search(r'\d+', x).group()))

    letters = [open(os.path.join(DIR, f)).read() for f in ordered_files]
    r = re.compile(r'\d')
    letters = [r.sub('', x) for x in letters]

    return letters, ordered_files 

def read_forum():
    import files
    csv = files.csv_reading('./forum-data-fixed/forum_post.csv', False, None)
    wrong_id =['91', '92', '144', '93', '129']
    wrong_topics = []
    topics_csv = files.csv_reading('./forum-data-fixed/forum_topic.csv', False, None)
    for i in topics_csv:
        if i[4] in wrong_id:
            wrong_topics.append(i[2])

    users_text = {}
    data = []
    users_text = {}
    for i in csv:
        if i[2] in wrong_topics:
            continue
        try:
            users_text[i[4]].append([i[1], i[3]])
        except:
            users_text[i[4]] = [[i[1], i[3]]]
        
    return users_text

def modeling(texts, mode='lda', num_words=10, num_topics=20, alpha='symmetric', dates=None, user=None):
    dictionary = corpora.Dictionary(texts)
    dictionary.save('./tmp/letters.dict')
    id2word = dictionary

    corpus = [id2word.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('./tmp/corpus.mm', corpus)
    corpus = corpora.MmCorpus('./tmp/corpus.mm')

    try:
        if mode == 'lda':
            model = models.ldamodel.LdaModel(
                corpus=corpus,
                id2word=id2word,
                num_topics=num_topics,
                alpha=alpha)
        elif mode == 'hdp':
            model = models.hdpmodel.HdpModel(alpha=0.1, gamma=0.1,
                corpus=corpus,
                id2word=id2word)
    except:
        return
    all_topics = model.show_topics(-1, num_words=num_words)


    all_topics = sorted(all_topics, key=lambda x: x[0])

    #find_topics_count(all_topics)

    # print(all_topics)

    topics = [model[doc] for doc in corpus]
    # print(topics)

    tfidf = models.TfidfModel(corpus)

    # with open('latter-topics.txt', 'w') as f:
    #     for x in zip(ordered_files, range(0, len(corpus))):
    #         f.write('{}: {}\n'.format(x[0], model[corpus[x[1]]]))

    # with open('topics.txt', 'w') as f:
    #     for top in all_topics:
    #         f.write('{}\n'.format(top))

    # topics = model[doc]
    # topics = [model[corpus[x]] for x in range(0, len(corpus))]

    # for x in topics:
    #     print(x)
    if dates == None:
        return all_topics, [model[x] for x in corpus]
        # num_topics_used = [len(model[doc]) for doc in corpus]
        # plt.hist(num_topics_used)
        #x = [i for i in range(1, len(dates) + 1)]
        #plt.xticks(x, dates, rotation='vertical')
        #plt.show()
    else:
        num_topics_used = [len(model[corpus[x]]) for x in range(len(corpus))]
        #plt.hist(num_topics_used)
        fig = plt.figure()
        x = [i for i in range(1, len(dates) + 1)]
        plt.plot(x, num_topics_used)
        plt.xticks(x, dates, rotation='vertical')
        url = 'users topics/user_{}.png'.format(user)
        try:
            plt.savefig(url)
        except:
            return
        #plt.show()

def popular_between_dates(forum, date_from, date_to):
    texts = []
    for user in forum:
        for text in forum[user]:
            d = text[0].split()
            d = d[0].split('-')
            try:
                user_date = datetime.datetime(int(d[0]), int(d[1]), int(d[2]))
            except ValueError:
                continue
            if date_from <= user_date <= date_to:
                texts.append(text[1])
    proces_text = processing_texts(texts)

    all_topics, topics = modeling(proces_text, 'hdp', num_topics=100, num_words=5, alpha=1)
    pops = [[x, 0] for x in range(len(all_topics))]
    for x in topics:
        for i in x:
            pops[i[0]][1] += 1
    pops = sorted(pops, key=lambda x: x[1], reverse=True)
    sizes = pops[:10]
    #labels = [all_topics[x[0]][1] for x in sizes]
    labels = range(1, len(sizes)+1)

    fig1, ax1 = plt.subplots()
    ax1.pie([x[1] for x in sizes], labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    labels = [all_topics[x[0]][1] for x in sizes]

    for x in labels:
        print("{}".format(x))
    plt.show()


def users_topics(forum):
    for user in forum:
        texts = []
        dates = []
        for text in forum[user]:
            d = text[0].split()
            try:
                find = dates.index(d[0])
            except ValueError:
                find = -1
            if find == -1:
                dates.append(d[0])
                texts.append(text[1])
            else:
                texts[find]+=" {}".format(text[1])

        if len(dates) < 3:
            continue
        proc_texts = processing_texts(texts)
        modeling(proc_texts, mode='hdp', alpha=1, dates=dates, user=user)

def found_topics_20news(method, amount=None, show=False, percent=None):
    folders = os.listdir('./20news-bydate/20news-bydate-test/')
    folders = folders[:10]

    ordered_files = []
    for folder in folders:
        test_dir = './20news-bydate/20news-bydate-test/{}/'.format(folder)
        train_dir = './20news-bydate/20news-bydate-train/{}/'.format(folder)

        for DIR in [test_dir, train_dir]:
            files = os.listdir(DIR)
            group_list = []
            for f_name in files:
                #with open(DIR + f_name, 'rb') as f:                
                with codecs.open(DIR + f_name, "r",encoding='utf-8', errors='ignore') as f:
                    ordered_files.append(f.read())

    proc_text = processing_texts(ordered_files)

    all_topics, topics = modeling(proc_text, mode=method, alpha=1, num_topics=200, num_words=10)
    pops = [[x, 0] for x in range(len(all_topics))]
    for x in topics:
        for i in x:
            pops[i[0]][1] += 1
    pops = sorted(pops, key=lambda x: x[1], reverse=True)
    if amount == None or amount > len(all_topics):
        sizes = pops[:len(all_topics)]
    else:
        sizes = pops[:amount]
    labels = [all_topics[x[0]][1] for x in sizes]

    if show == True:
        fig1, ax1 = plt.subplots()
        ax1.pie([x[1] for x in sizes], #labels=labels,
                autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        for x in labels:
            print("{}\n".format(x))
        print ("{}\n".format(len(labels)))
        plt.show()

    if percent != None:
        all_amount = sum([x[1] for x in pops])
        pops = [x[1] / all_amount for x in pops if (x[1] / all_amount) >= percent]
        print("most popular topics: {}\n".format(len(pops)))
        for x in labels[:len(pops)]:
            print("{}\n".format(x))
        print ("{}\n".format(len(labels)))

def main():    
    start_time = time.time()

    texts, ordered_files = read_letters()

    forum = read_forum()
    # listmerge=lambda ll: [el for lst in ll for el in lst]

    # found_topics_20news("lda", show=True, amount=20)
    # #users_topics(forum)

    popular_between_dates(forum,
                          datetime.datetime(2016, 12, 31),
                          datetime.datetime(2017, 1, 2))



if __name__ == '__main__':
    main()
