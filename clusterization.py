from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans

import nltk.stem
import os


class clusterization(object):

    def __init__(self):
        self.n_features = 0
        self.n_samples = 0
        self.vector = None
        self.vectorizer = None
        self.km = None
        self.lsa = None

    def clustering(self, n_clusters, init='k-means++', distanse=None):
        if distanse == 'cos':
            def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
                return cosine_distances(X, Y)
            k_means_.euclidean_distances = new_euclidean_distances

        self.km = KMeans(n_clusters=n_clusters, init=init,
                         max_iter=100, n_init=1, n_jobs=-1)
        try:
            self.km.fit(self.vector)
            return True
        except:
            return False

    def vectorizing(self, data, method='tfidf', lsa_n=None):
        
        self.lsa = lsa_n
        english_stemmer = nltk.stem.SnowballStemmer('english')
        if method == 'hashing':
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(stop_words='english', non_negative=True,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())

        elif method == 'tfidf':
            class StemmedTfidfVectorizer(TfidfVectorizer):
                def build_analyzer(self):
                    analyzer = super().build_analyzer()
                    return lambda doc: (
                        english_stemmer.stem(w) for w in analyzer(doc)
                    )

            vectorizer = StemmedTfidfVectorizer(max_df=0.5,
                                                min_df=2, stop_words='english')

        self.vector = vectorizer.fit_transform(data)

        self.n_features, self.n_samples = self.vector.shape

        if lsa_n:
            svd = TruncatedSVD(lsa_n)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)

            self.vector = lsa.fit_transform(self.vector)

    def estimation(self, real_labeling=None):

        if self.km:
            if real_labeling:
                homogenity = metrics.homogeneity_score(
                    real_labeling, self.km.labels_)
                completeness = metrics.completeness_score(
                    real_labeling, self.km.labels_)
                v_measure = metrics.v_measure_score(
                    real_labeling, self.km.labels_)

                return homogenity, completeness, v_measure
            if self.lsa:
                cal_har = metrics.calinski_harabaz_score(self.vector, self.km.labels_)
            else:
                cal_har = metrics.calinski_harabaz_score(self.vector.toarray(), self.km.labels_)


            return cal_har

        else:
            return False

    def write_clusters(self, files_name, data, DIR='./clusters'):

        if not os.path.exists(DIR):
            os.makedirs(DIR)

        for filename, letter, cluster in zip(files_name, data, self.km.labels_):
            dirname = "{}/{}".format(DIR, cluster)

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            with open("./{}/{}/{}".format(DIR, cluster, filename), 'w') as f:
                f.write(letter)
