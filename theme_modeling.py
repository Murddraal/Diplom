from collections import defaultdict
import nltk.stem
from gensim import corpora, models
import matplotlib.pyplot as plt
import processing_text as prct
import wordcloud
import files

# coding: utf8


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class topic_modeling(object):
    def __init__(self):
        self.proc_texts = None
        self.model = None
        self.corpus = None

    def texts_to_words(self, texts, stopwords_f=None):

        stoplist = []
        if stopwords_f:
            with open(stopwords_f, 'r') as f:
                stoplist = f.read().split(',')

        self.proc_texts = [prct.del_mails(text) for text in texts]
        self.proc_texts = [prct.del_urls(text) for text in texts]
        self.proc_texts = [
            prct.del_neither_num_nor_abc(text) for text in texts]

        english_stemmer = nltk.stem.SnowballStemmer('english')

        self.proc_texts = [[english_stemmer.stem(word) for word in latter.lower().split() if word not in stoplist and is_int(word) == False]
                           for latter in self.proc_texts]

        # frequency = defaultdict(int)
        # for latter in self.proc_texts:
        #     for token in latter:
        #         frequency[token] += 1

        # self.proc_texts = [[token for token in text if frequency[token] > 3 and frequency[token] < sum(frequency.items()) * 0.7]
        #                    for text in self.proc_texts]

    def modeling(self, mode='lda', num_topics=20, alpha='symmetric'):

        dictionary = corpora.Dictionary(self.proc_texts)
        # dictionary.save('./tmp/letters.dict')
        id2word = dictionary

        self.corpus = [id2word.doc2bow(text) for text in self.proc_texts]
        corpora.MmCorpus.serialize('./corpus.mm', self.corpus)
        self.corpus = corpora.MmCorpus('./corpus.mm')

        if mode == 'lda':
            self.model = models.ldamodel.LdaModel(
                corpus=self.corpus,
                id2word=id2word,
                num_topics=num_topics,
                alpha=alpha)
        elif mode == 'hdp':
            self.model = models.hdpmodel.HdpModel(alpha=0.1, gamma=0.1,
                                                    corpus=self.corpus,
                                                    id2word=id2word)


    def get_all_topics(self, num_words=10):

        all_topics = self.model.show_topics(-1, num_words=num_words)

        return sorted(all_topics, key=lambda x: x[0])

    def get_topics_for_each_text(self):
        return [self.model[doc] for doc in self.corpus]

    def circle_diagram(self, amount, num_words=10, show=False, DIR=None):

        all_topics = self.get_all_topics(num_words=num_words)
        topics = self.get_topics_for_each_text()

        pops = [[x, 0] for x in range(len(all_topics))]

        for x in topics:
            for i in x:
                pops[i[0]][1] += 1

        pops = sorted(pops, key=lambda x: x[1], reverse=True)

        if amount > len(all_topics):
            amount = len(all_topics)


        labels = range(1, amount+1)

        fig1, ax1 = plt.subplots()
        ax1.pie([x[1] for x in pops[:amount]], labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)

        ax1.axis('equal')

        for x in [all_topics[x[0]][1] for x in pops[:amount]]:
            print("{}".format(x))
        if show == True:
            plt.show()
        if DIR:
            plt.savefig(DIR)
##############################################
def read_forum():
    csv = files.csv_reading('./forum-data-fixed/forum_post.csv', False, None)
    wrong_id =['91', '92', '144', '93', '129']
    wrong_topics = []
    topics_csv = files.csv_reading('./forum-data-fixed/forum_topic.csv', False, None)
    for i in topics_csv:
        if i[4] in wrong_id:
            wrong_topics.append(i[2])

    users_text = {}
    data = []
    # for i in csv:
    #     try:
    #         users_text[i[4]].append([i[1], i[3]])
    #     except:
    #         users_text[i[4]] = [[i[1], i[3]]]
    for i in csv:
        if i[2] in wrong_topics:
            continue
        data.append(i[3])        
    return data

def read_letters():
    import os
    import re
    DIR = "./letters/translated"
    ordered_files = sorted(os.listdir(
        DIR), key=lambda x: int(re.search(r'\d+', x).group()))

    letters = [open(os.path.join(DIR, f)).read() for f in ordered_files]
    r = re.compile(r'\d')
    letters = [r.sub('', x) for x in letters]

    return letters 

data = read_forum()

tm = topic_modeling()
tm.texts_to_words(data, stopwords_f='stopwords.txt')
tm.modeling('hdp', num_topics=10)
tm.circle_diagram(10, num_words=5, show=True)