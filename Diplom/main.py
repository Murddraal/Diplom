import requests
import json

def text_parsing(json_str):
    json_str = """ 
    {
    "_id":{"$oid":"584e766138d936063bc84f8f"},
    "messageId":"158e5e6bd44a3ad7",
    "threadId":"158e5e6bd44a3ad7",
    "historyId":"78436",
    "subject":"Fwd: My email was already used?",
    "snippet":"Miss Olga... it was just easier to forward this to you. Cha Please visit our website and also our Official Facebook Groups for more information and also to join in discussions. https://www.facebook.com",
    "bodyText":"",
    "receivedAt":1.481325852e+12,
    "labels":["CATEGORY_PERSONAL","INBOX"],
    "__v":0
    }
    """
    mail = json.loads(json_str)

    return mail['snippet']


    

url = 'https://translate.yandex.net/api/v1.5/tr.json/translate?'
# ключ для подключения к API
key = 'trnsl.1.1.20170127T090047Z.d0549b75a2237806.40d7e9d8902cb56bc4115af8e88c1bf8f0e4cbf3'
text = 'Плывущий во мраке ночи увалень'
lang = 'en'
text = text_parsing("")
translation = requests.post(url, data={'key': key, 'text': text, 'lang': lang})
# Выводим результат
jtr = json.loads(translation.text)

print(jtr['text'][0])
