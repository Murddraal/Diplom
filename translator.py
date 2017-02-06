import requests
import json
import csv
import re
from docx import Document
# coding: utf8


class translator(object):
    def __init__(self, lang='en'):
        self.url = 'https://translate.yandex.net/api/v1.5/tr.json/translate?'
        self.key = 'trnsl.1.1.20170127T090047Z.d0549b75a2237806.40d7e9d8902cb56bc4115af8e88c1bf8f0e4cbf3'
        self.lang = lang

    def csv_reading(self, csv_filename):
        with open(csv_filename, encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, dialect='excel-tab')
            for row in reader:
                str_row = ""
                for cell in row:
                    str_row += cell
                yield str_row

    def writing(self, file_name, orig_text, trans_text):
        try:
            document = Document(file_name)
        except:
            document = Document()

        table = document.add_table(rows=1, cols=2)
        row = table.rows[0]
        row.cells[0].text, row.cells[1].text = orig_text, trans_text
        document.save(file_name)

    def text_parsing(self, json_row):
        mail = json.loads(json_row)

        return mail['bodyText']

    def translating(self, text):

        splited_text = re.split(r'[?!\.]', text)

        translated = ""
        for i in splited_text:
            if len(i) > 10000:
                return translated
            translation = requests.post(self.url,
                                        data={'key': self.key,
                                              'text': i,
                                              'lang': self.lang}
                                        )

            jtr = json.loads(translation.text)
            translated += jtr['text'][0] + " "
        return translated
