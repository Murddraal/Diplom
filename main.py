"""main moduleof translation programm
"""

import time
import translator as tr
from my_wordcloud import My_wordcloud as mwc

# coding: utf8


def main():
    """logic of the programm"""
    transl = tr.Translator('en')
    csv = tr.csv_reading('emails.csv')
    k = 1

    start_time = time.time()

    for i in csv:
        k += 1
        text = tr.text_parsing(i)
        cloud = mwc()
        cloud.gen_cloud('./clouds/cloud{}.png'.format(k),
                        './clouds/cloud{}.txt'.format(k), text, True)

        transl.lang_detecting(text, 0)
        transl.translating(text)
        if k == 100:
            break

    transl.print_langs_list("langs.txt")

    total_time = time.time() - start_time

if __name__ == '__main__':
    main()
