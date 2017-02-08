import requests
import json
import csv
import re
from docx import Document
from docx.shared import Pt
# coding: utf8


class translator(object):
    def __init__(self, lang='en'):
        self.url = 'https://translate.yandex.net/api/v1.5/tr.json/translate?'
        self.key = 'trnsl.1.1.20170127T090047Z.d0549b75a2237806.40d7e9d8902cb56bc4115af8e88c1bf8f0e4cbf3'
        self.lang = lang
        self.max_text_size = 10000

    def csv_reading(self, csv_filename):
        with open(csv_filename, encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, dialect='excel-tab')
            for row in reader:
                str_row = ""
                for cell in row:
                    str_row += cell
                yield str_row

    def writing(self, orig_text, trans_text):
        try:
            tr_document = Document('translation.docx')
        except:
            tr_document = Document()
        try:
            orig_document = Document('original.docx')
        except:
            orig_document = Document()

        run = tr_document.add_paragraph().add_run()
        font = run.font
        font.name, font.size = 'Times New Roman', Pt(6)
        table = tr_document.add_table(rows=1, cols=1)
        row = table.rows[0]
        row.cells[0].text = trans_text
        tr_document.save('translation.docx')

        run = orig_document.add_paragraph().add_run()
        font = run.font
        font.name, font.size = 'Times New Roman', Pt(6)
        table = orig_document.add_table(rows=1, cols=1)
        row = table.rows[0]
        row.cells[0].text = orig_text
        orig_document.save('original.docx')

    def text_parsing(self, json_row):
        mail = json.loads(json_row)

        return mail['bodyText']

    def translating(self, text):

        splited_text = re.split(r'[?!:\n\.]', text)

        translated = ""
        for i in splited_text:
            if len(i) > self.max_text_size:
                return translated
            if len(re.findall(r'[^\s?1\w\n]', i)) == 0:
                continue
            translation = requests.post(self.url,
                                        data={'key': self.key,
                                              'text': i,
                                              'lang': self.lang}
                                        )

            jtr = json.loads(translation.text)
            translated += jtr['text'][0] + " ||| "
        return translated
