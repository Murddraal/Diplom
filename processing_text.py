"""This module translate text, detect languages.
"""

import json
import re
import requests
from langdetect import detect_langs
from langdetect import lang_detect_exception as ld_exc
import nltk
from collections import defaultdict

# coding: utf8

# DetectorFactory.seed = 0


class Translator(object):
    """This class can translate text via yandex.translate service,
    detect language via langdetect and yandex translate.
    """

    def __init__(self, url_tr, url_det, key, lang='en'):
        self.url_tr = 'https://translate.yandex.net/api/v1.5/tr.json/translate?'
        self.url_det = 'https://translate.yandex.net/api/v1.5/tr.json/detect? '
        self.key = 'trnsl.1.1.20161229T092652Z.92eec6478cc791c5.9a428e7546269ce12b04df552805c48a4c63261e'
        self.lang = lang
        self.max_text_size = 10000
        self.languages = {}

    def translating(self, text):
        """translate text via yandex.translate
        """
        splited_text = re.split(r'[?!:\n\.]', text)

        translated = ""
        for i in splited_text:
            if len(i) > self.max_text_size:
                return translated
            if len(re.findall(r'[^\s?1\w\n]', i)) == 0:
                continue
            translation = requests.post(self.url_tr,
                                        data={'key': self.key,
                                              'text': i,
                                              'lang': self.lang})

            jtr = json.loads(translation.text)
            translated += jtr['text'][0] + " ||| "
        return translated

    def lang_detecting(self, text):
        """detect language via langdetect
        """

        try:
            lg = detect_langs(text)
        except TypeError:
            return
        except ld_exc.LangDetectException:
            return

        list_lg = [str(i).split(':') for i in lg]

        for i in list_lg:
            try:
                self.languages[i[0]] += 1
            except KeyError:
                self.languages[i[0]] = 1

    def get_langs_list(self):
        """return all detected language with it's weight
        """
        return self.languages


def del_mails(text):
    return re.sub(r'\S+@\S+', '', text)
    
def del_urls(text):
    text = re.sub(r'\S+\.\S+', '', text)
    return re.sub(r'http\S+', '', text)

def del_neither_num_nor_abc(text):
    return re.sub(r'[^a-zA-Z0-9_\s]', '', text)
