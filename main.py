"""main moduleof translation programm
"""

import time
import translator as tr
import re
from my_wordcloud import My_wordcloud as mwc

# coding: utf8


def main():
    """logic of the programm"""
    transl = tr.Translator('en')
    csv = tr.csv_reading('emails.csv')
    k = 1

    text_for_cloud = ""

    start_time = time.time()


    def deleting_sites(text):
        k = re.split(r'\s\S*\.com\S*', text)
        t = ""
        for i in k:
            t += i + ' '
        return t


    for i in csv:
        k += 1
        parsed_text = tr.text_parsing(i)
        lg = transl.lang_detecting(parsed_text, 0)

        parsed_text = deleting_sites(parsed_text)

        #if not(len(lg) == 1 and lg[0].lang == 'en'):
        text_for_cloud += transl.translating(parsed_text) + " "
        #else:
            #text_for_cloud += parsed_text
        print(k)
        if k == 2:
            break

    cloud = mwc()
    cloud.gen_cloud('./clouds/cloud{}.png'.format(k),
                    './clouds/cloud{}.txt'.format(k), text_for_cloud, True)
    transl.print_langs_list("langs.txt")

    total_time = time.time() - start_time

if __name__ == '__main__':
    main()
