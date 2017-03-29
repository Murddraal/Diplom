"""contains class wich work with wordcloud
"""

# -*- coding: utf8 -*-

import operator
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class My_wordcloud(object):
    def gen_cloud(self, picname, wordsname, text, if_sw=False):
        stopwords = ""
        if if_sw:
            with open("stopwords.txt", "r") as file:
                stopwords = file.read()
            stopwords = stopwords.split(",")

        try:
            wordcloud = WordCloud(max_words=1000, max_font_size=40,
                                  collocations=True, stopwords=stopwords).generate(text)
        except ValueError:
            return False

        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(picname)
        plt.close()

        words = wordcloud.words_
        sorted_words = sorted(words.items(), key=operator.itemgetter(1))
        sorted_words.reverse()
        with open(wordsname, 'w') as file:
            for j in sorted_words:
                file.write(j[0] + ' - ' + str(j[1]) + '\n')

        return True
