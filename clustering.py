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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction import text
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from random import randint
from sklearn.feature_extraction.text import CountVectorizer
import codecs
import LDA
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline



# coding: utf8


def draw_result(groups, clusters):

    import pylab as pl
    import matplotlib.pyplot as plt

    ind = np.arange(len(clusters))
    width = 0.1
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, groups, width, color='r')

    rects2 = ax.bar(ind + width, clusters, width, color='y')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(['{}'.format(x) for x in range(1, len(clusters) + 1)])

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.show()


def writing_in_clusters(ordered_files, letters, labels_):

    clusters_items = [(x, y, str(z))
                      for x, y, z in zip(ordered_files, letters, labels_)]

    if os.path.exists("./clusters"):
        shutil.rmtree("./clusters")
    if not os.path.exists("./clusters"):
        os.makedirs("./clusters")
    for filename, letter, dname in clusters_items:
        dirname = "./clusters/{}".format(dname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open("./clusters/{}/{}".format(dname, filename), 'w') as f:
            f.write(letter)


def clusterisation(data, clust_numb, n_components=None):
    english_stemmer = nltk.stem.SnowballStemmer('english')

    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super().build_analyzer()
            return lambda doc: (
                english_stemmer.stem(w) for w in analyzer(doc)
            )

    stop_words = text.ENGLISH_STOP_WORDS.union(
        [x for x in open("stopwords.txt", 'r').read().split(',')])

    vectorizer = StemmedTfidfVectorizer(
        min_df=2, max_df=0.5, stop_words=stop_words, decode_error='ignore')

    vectorized = vectorizer.fit_transform(data)

    if n_components:
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        vectorized = lsa.fit_transform(vectorized)
        explained_variance = svd.explained_variance_ratio_.sum()


    # def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    #     return cosine_similarity(X, Y)

    # monkey patch (ensure cosine dist function is used)

    #k_means_.euclidean_distances = new_euclidean_distances

    km = KMeans(n_clusters=clust_numb, init='k-means++',
                n_init=1, n_jobs=1)
    km.fit(vectorized)

    # print("Top terms per cluster:")

    # order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    # for i in range(clust_numb):
    #     print("Cluster %d:" % i, end='')
    #     for ind in order_centroids[i, :10]:
    #         print(' %s' % terms[ind], end='')
    #     print()
    #print("{}".format([len([x for x in km.labels_ if x == k]) for k in range(clust_numb)]))

    return vectorized,  km


def clust_test(clust_numb, files, iterations):

    data = [[x[1] for x in y if x[0].find('train') != -1] for y in files]

    train_lenghtes = [len(x) for x in data]

    def listmerge(ll): return [el for lst in ll for el in lst]

    data = listmerge(data)

    v, train_km = clusterisation(data, clust_numb)

    clust_sizes = [len([x for x in train_km.labels_ if x == k])
                   for k in range(clust_numb)]

    for i in range(iterations):
        data = []
        used_index = []
        for train_length, f in zip(train_lenghtes, files):
            tr_data = []
            for j in range(train_length):
                index = randint(0, train_length - 1)
                while (index in used_index):
                    index = randint(0, train_length - 1)
                tr_data.append(f[index][1])
            data.append(tr_data)

        train_km = clusterisation(listmerge(data), clust_numb)
        clust_sizes = [x + y for x, y in zip(clust_sizes,
                                             [len([x for x in train_km.labels_ if x == k])
                                              for k in range(clust_numb)])]

    clust_sizes = [x / iterations for x in clust_sizes]

    print('{}'.format(clust_sizes))

    return train_lenghtes, clust_sizes
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

def main():
    start_time = time.time()
    DIR = "./letters/translated"
    ordered_files = sorted(os.listdir(
        DIR), key=lambda x: int(re.search(r'\d+', x).group()))

    letters = [open(os.path.join(DIR, f)).read() for f in ordered_files]
    r = re.compile(r'\d')
    letters = [r.sub('', x) for x in letters]

    forum = read_forum()
    text = ""
    for i in forum:
        text += "{} ".format(i)

    import my_wordcloud as mwc

    wc = mwc.My_wordcloud()
    wc.gen_cloud('forums_wc.png', 'forum_wc.txt', text, True)

    # folders = os.listdir('./20news-bydate/20news-bydate-test/')
    # folders = folders[:10]

    # ordered_files = []
    # for folder in folders:
    #     test_dir = './20news-bydate/20news-bydate-test/{}/'.format(folder)
    #     train_dir = './20news-bydate/20news-bydate-train/{}/'.format(folder)

    #     for DIR in [test_dir, train_dir]:
    #         files = os.listdir(DIR)
    #         group_list = []
    #         for f_name in files:
    #             with codecs.open(DIR + f_name, "r",encoding='utf-8', errors='ignore') as f:
    #                 ordered_files.append(f.read())
        #ordered_files.append(group_list)


    # with open("features_list.txt", 'w') as f:
    #     for term in terms:
    #         f.write("{}\n".format(term))

    #num_clusters = len(folders)
    # from sklearn import metrics

    # mks = [0 for x in range(9)]

    # for j in range(1):
    #     for i in range(2, 11): 
    #         X, km = clusterisation(letters, i)
    #         mks[i-2] += metrics.calinski_harabaz_score(X.toarray(), km.labels_)
            
    
    # #mks = [x/10 for x in mks]
    # for i in mks:
    #     print("{}".format(i))

    # #groups, clusters = clust_test(num_clusters, ordered_files, 4)
    # #draw_result(groups, clusters)
    # # from sklearn.cluster import KMeans

    # # from sklearn.metrics.pairwise import cosine_distances
    # # def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    # #     return cosine_distances(X,Y)

    # # # monkey patch (ensure cosine dist function is used)
    # # from sklearn.cluster import k_means_

    # # k_means_.euclidean_distances = new_euclidean_distances
    # # km = KMeans(n_clusters=num_clusters, init='random',
    # #             n_init=10, n_jobs=-1)
    # # km.fit(vectorized)

    # #writing_in_clusters(ordered_files, letters, km.labels_)

    # total_time = time.time() - start_time

    # print("Time: {}".format(total_time))

    # # draw_result(vectorized, num_clusters, km.labels_)

    # # centers = [x for x in km.cluster_centers_]

    # # nearest_letter = []

    # # all = [(x[0], x[1], v, z)
    # # for x, v, z in zip(ordered_files, vectors, km.labels_) if len(z.data) !=
    # # 0]

    # # with open('vectors.txt', 'w') as f:
    # #     for vec in vectors:
    # #         for x in vec.data:
    # #             f.write("{} ".format(x))
    # #         f.write("\n\n")

    # # cluster_index = 0

    # # for vec in centers:
    # #     letter = ""
    # #     min_dist = 100000
    # #     for i in all:
    # #         if cluster_index == i[3]:
    # #             dist = np.linalg.norm(vec - i[2])
    # #             if dist < min_dist:
    # #                 min_dist = dist
    # #                 letter = i[0]
    # #     cluster_index += 1

    # #     nearest_letter.append(letter)

    # # with open("nearest_letters.txt", 'w') as f:
    # #     k = 1
    # #     for letter in nearest_letter:
    # #         f.write("cluster {}:\n{}\n\n".format(k, letter))
    # #         k += 1


if __name__ == '__main__':
    main()
