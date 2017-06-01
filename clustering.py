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

# coding: utf8


def draw_result(groups, vectorized, clust_number, labels):

    # from sklearn.decomposition import TruncatedSVD as TSVD
    # tsvd = TSVD(n_components=2, algorithm="randomized").fit(vectorized)
    # tsvd_2d = tsvd.transform(vectorized)

    import pylab as pl
    import matplotlib.pyplot as plt

    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    # markers = ['8', 'o', '*', 'v', '^', '<', '>',
    #            's', 'p', 'h', 'H', 'D', 'd', 'P']

    # for i in range(0, tsvd_2d.shape[0]):
    #     for j in range(clust_number):
    #         if labels[i] == j:
    #             pl.scatter(tsvd_2d[i, 0], tsvd_2d[i, 1], c=colors[j],
    #                        marker=markers[j])

    # pl.title(
    #     'Mails dataset with {} clusters'.format(clust_number))
    # pl.show()

    real = [x[1] for x in groups]

    ind = np.arange(clust_number)
    width = 0.1
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, real, width, color='r')

    clustered = [len([x for x in labels if x == i]) for i in range(clust_number)]

    rects2 = ax.bar(ind + width, clustered, width, color='y')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels((x[0] for x in groups))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
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


def main():

    start_time = time.time()
    # DIR = "./letters/translated"
    # ordered_files = sorted(os.listdir(
    #     DIR), key=lambda x: int(re.search(r'\d+', x).group()))

    # letters = [open(os.path.join(DIR, f)).read() for f in ordered_files]
    # r = re.compile(r'\d')
    # letters = [r.sub('', x) for x in letters]

    import sklearn.datasets
    groups = ['comp.graphics',
              'sci.space',
              'talk.politics.guns',
              'alt.atheism',
              'rec.autos',
              'soc.religion.christian',
              'sci.med']
    test_data = sklearn.datasets.fetch_20newsgroups(data_home='./', subset='train', categories=groups)

    groups = [('comp.graphics', 584),
              ('sci.space', 593),
              ('talk.politics.guns', 546),
              ('alt.atheism', 480),
              ('rec.autos', 594),
              ('soc.religion.christian',599),
              ('sci.med', 594)]

    english_stemmer = nltk.stem.SnowballStemmer('english')
    from sklearn.feature_extraction.text import TfidfVectorizer

    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super().build_analyzer()
            return lambda doc: (
                english_stemmer.stem(w) for w in analyzer(doc)
            )

    from sklearn.feature_extraction import text

    stop_words = text.ENGLISH_STOP_WORDS.union(
        [x for x in open("stopwords.txt", 'r').read().split(',')])

    vectorizer = StemmedTfidfVectorizer(
        min_df=10, max_df=0.5, stop_words=stop_words, decode_error='ignore')
    vectorized = vectorizer.fit_transform(test_data.data)
    #vectors = [x for x in vectorized if len(x.data) != 0]
    vectors = [x for x in vectorized]

    num_samples, num_features = vectorized.shape
    print("samples: {}\nfeatures: {}".format(num_samples, num_features))
    terms = vectorizer.get_feature_names()

    with open("features_list.txt", 'w') as f:
        for term in terms:
            f.write("{}\n".format(term))

    num_clusters = len(groups)
    from sklearn.cluster import KMeans

    from sklearn.metrics.pairwise import cosine_distances
    def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
        return cosine_distances(X,Y)

    # monkey patch (ensure cosine dist function is used)
    from sklearn.cluster import k_means_

    k_means_.euclidean_distances = new_euclidean_distances 
    km = KMeans(n_clusters=num_clusters, init='random',
                n_init=10, n_jobs=-1)
    km.fit(vectorized)

    #writing_in_clusters(ordered_files, letters, km.labels_)

    total_time = time.time() - start_time

    print("Time: {}".format(total_time))

    draw_result(groups, vectorized, num_clusters, km.labels_)

    centers = [x for x in km.cluster_centers_]

    nearest_letter = []

    all = [(x, y, z, k, len(z.data))
           for x, y, z, k in zip(ordered_files, letters, vectors, km.labels_) if len(z.data) != 0] 
    empty_vect = all[0][2]
    non_empty_vect = all[1][2]

    with open('vectors.txt', 'w') as f:
        for vec in vectors:
            for x in vec.data:
                f.write("{} ".format(x))
            f.write("\n\n")

    cluster_index = 0

    for vec in centers:
        letter = ""
        min_dist = 100000
        for i in all:
            if cluster_index == i[3]:
                dist = numpy.linalg.norm(vec - i[2])
                if dist < min_dist:
                    min_dist = dist
                    letter = i[0]
        cluster_index += 1

        nearest_letter.append(letter)

    with open("nearest_letters.txt", 'w') as f:
        k = 1
        for letter in nearest_letter:
            f.write("cluster {}:\n{}\n\n".format(k, letter))
            k += 1


if __name__ == '__main__':
    main()
