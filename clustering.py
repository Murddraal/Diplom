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


def draw_result(vectorized, clust_number, labels):

    from sklearn.decomposition import TruncatedSVD as TSVD
    tsvd = TSVD(n_components=clust_number, algorithm="arpack").fit(vectorized)
    tsvd_2d = tsvd.transform(vectorized)

    import pylab as pl
    import matplotlib as mpl
    colors = ['k', 'k', 'b', 'c', 'm', 'y', 'k']
    markers = ['o', 'o', '*', 'v', '^', '<', '>', '8',
               's', 'p', 'h', 'H', 'D', 'd', 'P', 'X']

    for i in range(0, tsvd_2d.shape[0]):
        for j in range(clust_number):
            if labels[i] == j:
                pl.scatter(tsvd_2d[i, 0], tsvd_2d[i, 1], c=colors[j],
                           marker=markers[j])

    pl.title(
        'Mails dataset with {} clusters and known outcomes'.format(clust_number))
    pl.show()


def writing_in_clusters(ordered_files, latters, labels_):

    clusters_items = [(x, y, str(z))
                      for x, y, z in zip(ordered_files, latters, labels_)]

    if os.path.exists("./clusters"):
        shutil.rmtree("./clusters")
    if not os.path.exists("./clusters"):
        os.makedirs("./clusters")
    for filename, latter, dname in clusters_items:
        dirname = "./clusters/{}".format(dname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open("./clusters/{}/{}".format(dname, filename), 'w') as f:
            f.write(latter)


def main():

    start_time = time.time()
    DIR = "./latters/translated"
    ordered_files = sorted(os.listdir(
        DIR), key=lambda x: int(re.search(r'\d+', x).group()))

    latters = [open(os.path.join(DIR, f)).read() for f in ordered_files]
    r = re.compile(r'\d')
    latters = [r.sub('', x) for x in latters]

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
        [x for x in open("clust_stopwords.txt", 'r').read().split('\n')])

    vectorizer = StemmedTfidfVectorizer(
        min_df=10, max_df=0.5, stop_words=stop_words, decode_error='ignore')
    vectorized = vectorizer.fit_transform(latters)
    #vectors = [x for x in vectorized if len(x.data) != 0]
    vectors = [x for x in vectorized]

    num_samples, num_features = vectorized.shape
    print("samples: {}\nfeatures: {}".format(num_samples, num_features))
    terms = vectorizer.get_feature_names()

    with open("features_list.txt", 'w') as f:
        for term in terms:
            f.write("{}\n".format(term))

    num_clusters = 5
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=num_clusters, init='random',
                n_init=10, verbose=1, n_jobs=1)
    km.fit(vectorized)

    writing_in_clusters(ordered_files, latters, km.labels_)

    total_time = time.time() - start_time

    print("Time: {}".format(total_time))

    #draw_result(vectorized, num_clusters, km.labels_)

    centers = [x for x in km.cluster_centers_]

    nearest_latter = []

    all = [(x, y, z, k, len(z.data))
           for x, y, z, k in zip(ordered_files, latters, vectors, km.labels_) if len(z.data) != 0] 
    empty_vect = all[0][2]
    non_empty_vect = all[1][2]

    with open('vectors.txt', 'w') as f:
        for vec in vectors:
            for x in vec.data:
                f.write("{} ".format(x))
            f.write("\n\n")

    cluster_index = 0

    for vec in centers:
        latter = ""
        min_dist = 100000
        for i in all:
            if cluster_index == i[3]:
                dist = numpy.linalg.norm(vec - i[2])
                if dist < min_dist:
                    min_dist = dist
                    latter = i[0]
        cluster_index += 1

        nearest_latter.append(latter)

    with open("nearest_latters.txt", 'w') as f:
        k = 1
        for latter in nearest_latter:
            f.write("cluster {}:\n{}\n\n".format(k, latter))
            k += 1


if __name__ == '__main__':
    main()
