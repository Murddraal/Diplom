"""
=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the scikit-learn can be used to cluster
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two feature extraction methods can be used in this example:

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

  - HashingVectorizer hashes word occurrences to a fixed dimensional space,
    possibly with collisions. The word count vectors are then normalized to
    each have l2-norm equal to one (projected to the euclidean unit-ball) which
    seems to be important for k-means to work in high dimensional space.

    HashingVectorizer does not provide IDF weighting as this is a stateless
    model (the fit method does nothing). When IDF weighting is needed it can
    be added by pipelining its output to a TfidfTransformer instance.

Two algorithms are demoed: ordinary k-means and its more scalable cousin
minibatch k-means.

Additionally, latent semantic analysis can also be used to reduce dimensionality
and discover latent patterns in the data. 

It can be noted that k-means (and minibatch k-means) are very sensitive to
feature scaling and that in this case the IDF weighting helps improve the
quality of the clustering by quite a lot as measured against the "ground truth"
provided by the class label assignments of the 20 newsgroups dataset.

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seem to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as text data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import nltk.stem


def vectorizing(opts, dataset):
    english_stemmer = nltk.stem.SnowballStemmer('english')
    svd = None
    if opts.use_hashing:
       
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(#n_features=opts.n_features,
                                    stop_words='english', non_negative=True,
                                    norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                        stop_words='english',
                                        non_negative=False, norm='l2',
                                        binary=False)
    else:
        class StemmedTfidfVectorizer(TfidfVectorizer):
            def build_analyzer(self):
                analyzer = super().build_analyzer()
                return lambda doc: (
                    english_stemmer.stem(w) for w in analyzer(doc)
                )

        vectorizer = StemmedTfidfVectorizer(max_df=0.5, #max_features=opts.n_features,
                                    min_df=2, stop_words='english',
                                    use_idf=opts.use_idf)


    X = vectorizer.fit_transform(dataset.data)

    #print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    #print()

    if opts.n_components:
        #print("Performing dimensionality reduction using LSA")
        #t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        #print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        # print("Explained variance of the SVD step: {}%".format(
        #     int(explained_variance * 100)))

        #print()

    return X, vectorizer, svd

def parsing_arguments():
    # parse commandline arguments
    op = OptionParser()
    op.add_option("--iterations",
                dest="iterations", default = 1, type="int")
    op.add_option("--lsa",
                dest="n_components", type="int",
                help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch",
                action="store_false", dest="minibatch", default=True,
                help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                action="store_false", dest="use_idf", default=True,
                help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                action="store_true", default=False,
                help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                help="Maximum number of features (dimensions)"
                    " to extract from text.")
    op.add_option("--verbose",
                action="store_true", dest="verbose", default=False,
                help="Print progress reports inside k-means algorithm.")

    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    return opts



def clustering(opts, true_k, X, init='k-means++'):
    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init=init, n_init=1,
                            init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        from sklearn.cluster import k_means_
        from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
        def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
            return cosine_distances(X, Y)

        # monkey patch (ensure cosine dist function is used)

        k_means_.euclidean_distances = new_euclidean_distances
        km = KMeans(n_clusters=true_k, init=init, max_iter=100, n_init=1,
                    verbose=opts.verbose, n_jobs=-1)

    #print("Clustering sparse data with %s" % km)
    #t0 = time()
    km.fit(X)
    return km
def main():
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    #print(__doc__)
    #op.print_help()

    opts = parsing_arguments()

    aver_metr = [0, 0, 0, 0, 0]
    perc = 0
    t0 = time()

    #for i in range(opts.iterations):
    for ii in range(2,21):
        t1 = time()
        ###############################################################################
        # Load some categories from the training set
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space',
        ]

        categories = None

        # print("Loading 20 newsgroups dataset for categories:")
        # print(categories)

############Testing##############################
        dataset = fetch_20newsgroups(subset='test', categories=categories,
                                    shuffle=True)
        labels = dataset.target
        true_k = np.unique(labels).shape[0]
        X, vectorizer, svd = vectorizing(opts, dataset)
        km = clustering(opts, ii, X)
        
        #print("% done in %0.3fs" % (i, time() - t1))

        perc_loc = 0
        for i in zip(labels, km.labels_):
            if i[0] == i[1]:
                perc_loc += 1
        perc_loc /= len(labels)
        perc += perc_loc

        if not opts.use_hashing:
            #print("Top terms per cluster:")

            if opts.n_components:
                original_space_centroids = svd.inverse_transform(km.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = km.cluster_centers_.argsort()[:, ::-1]

            terms = vectorizer.get_feature_names()
            # for i in range(true_k):
            #     print("Cluster %d:" % i, end='')
            #     for ind in order_centroids[i, :10]:
            #         print(' %s' % terms[ind], end='')
            #     print()
            #print("{}".format([len([x for x in km.labels_ if x == k]) for k in range(len(categories))]))

        aver_metr[0] += metrics.homogeneity_score(labels, km.labels_)
        aver_metr[1] += metrics.completeness_score(labels, km.labels_)
        aver_metr[2] += metrics.v_measure_score(labels, km.labels_)
        aver_metr[3] += metrics.adjusted_rand_score(labels, km.labels_)
        aver_metr[4] += metrics.silhouette_score(X, km.labels_, sample_size=1000)

        mk = metrics.calinski_harabaz_score(X, km.labels_)
        print("{} {}".format(ii, mk))


    aver_metr = [x / opts.iterations for x in aver_metr]

    # with open('measure_with_diff_lsa.txt', 'w') as f:
    #     f.write("Homogeneity: %0.3f" % aver_metr[0])
    #     f.write("Completeness: %0.3f" % aver_metr[1])
    #     f.write("V-measure: %0.3f" % aver_metr[2])

    mk = metrics.calinski_harabaz_score(X, km.labels_)
    print(mk)
    # print()
    # print("done in %0.3fs" % (time() - t0))

if __name__ == '__main__':
    main()
