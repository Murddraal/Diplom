# coding: utf8
import matplotlib.pyplot as plt
import operator
from wordcloud import WordCloud


def test_collocations():
    text = ""
    with open('./test_text.txt', 'r') as f:
        text = f.read()

    stopwords = ""
    with open("stopwords.txt", "r") as file:
        stopwords = file.read()
        stopwords = stopwords.split(",")

    wordcloud = WordCloud(max_words=1000, max_font_size=40,
                          collocations=True, stopwords=stopwords)
    wordcloud.generate(text)
    #wordcloud = WordCloud(max_font_size=40, collocations=True, max_words = 1000).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    url = './tests/{}.png'.format(text[:10])
    plt.savefig(url)

    words = wordcloud.words_
    sorted_words = sorted(words.items(), key=operator.itemgetter(1))
    sorted_words.reverse()

    with open('./tests/{}.txt'.format(text[:10]), 'w') as f:
        for j in sorted_words:
            f.write('{} - {}\n'.format(str(j[0]), str(j[1])))


test_collocations()
