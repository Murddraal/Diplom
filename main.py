import operator
import time
import translator as tr
from wordcloud import WordCloud
from os import path
import matplotlib.pyplot as plt
# coding: utf8

def main():
    test = tr.translator('en')
    x = test.csv_reading('emails.csv')
    k = 1
    start_time = time.time()
    for i in x:
        #print(k)
        k += 1
        text = test.text_parsing(i)
        if text != "":
            wordcloud = WordCloud().generate(text)
            plt.imshow(wordcloud, interpolation = 'bilinear')
            plt.axis("off")
            wordcloud = WordCloud(max_font_size=40).generate(text)
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            url = './clouds/cloud{}.png'.format(k)
            plt.savefig(url)

            words = wordcloud.words_
            sorted_words = sorted(words.items(), key = operator.itemgetter(1))
            sorted_words.reverse()
            with open('./clouds/cloud{}.txt'.format(k), 'w') as f:
                for j in sorted_words:
                    f.write('{} - {}\n'.format(j[0], j[1]))

        # tr_text = test.translating(text)
        # test.writing(text, tr_text)
        # print("wr")
        langs = test.lang_detecting(text, 0)
        if k == 5:
            break
    total_time = time.time() - start_time

    sorted_langs = sorted(langs.items(), key=operator.itemgetter(1), reverse=False)

    sorted_langs.reverse()
    with open('langs_ya.txt', 'w') as f:
        f.write('Time: ' + str(total_time) + '\n\n')
        for i in sorted_langs:
            f.write(str(i[0]) + '-' + str(i[1]) + '\n')

if __name__ == '__main__':
    main()
