"""contains class wich work with wordcloud
"""

# -*- coding: utf8 -*-

import operator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

class My_wordcloud(object):
    def gen_cloud(self, picname, wordsname, text, sw=None):
        stopwords = ""
        if sw:
            with open(sw, "r") as file:
                stopwords = file.read()
            stopwords = stopwords.split(",")
        
        english_stemmer = nltk.stem.SnowballStemmer('english')

        text = [english_stemmer.stem(word) for word in text.lower().split() if word not in stopwords and is_int(word) == False ]
        
        text = " ".join(text)

        try:
            wordcloud = WordCloud(max_words=1000, max_font_size=40,
                                  collocations=True, stopwords=stopwords).generate(text)
        except ValueError:
            return False

        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(picname)
        #plt.show()
        plt.close()

        words = wordcloud.words_
        sorted_words = sorted(words.items(), key=operator.itemgetter(1))
        sorted_words.reverse()
        with open(wordsname, 'w') as file:
            for j in sorted_words:
                file.write(j[0] + ' - ' + str(j[1]) + '\n')

        return True
