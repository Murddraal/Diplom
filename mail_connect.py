# -*- coding: utf-8 -*-
import imaplib
import email
import os, time

#Имя и пароль для соединения с электронной почтой
username = 'ursavickiy94@gmail.com'
password = 'murd14121967'

#Создаём файл для записи сообщений(Название файла: текущее время)
file_name = 'Date'+time.strftime("%d") + '.' + time.strftime("%m") + '.' + time.strftime("%Y") + '_Time' + time.strftime("%H") + '-' + time.strftime("%M") + '-' + time.strftime("%S")
f_message = open(file_name+".txt", 'w')

#Соединяемся с сервером gmail.com через imap4_ssl
gmail = imaplib.IMAP4_SSL('imap.gmail.com')
gmail.login(username, password)
gmail.list()

#Выбирает из папки входящие непрочитанные сообщения
typ, count = gmail.select('inbox')

#Выводит количество непрочитанных сообщений в папке входящие
typ, unseen = gmail.status('inbox', "(UNSEEN)")
#print unseen

#Главный блок
k = 0
typ, data = gmail.search(None, '(SEEN)')

ids = data[0].split()

latest_email_id = ids[-1]

result, data = gmail.fetch(latest_email_id, "(RFC822)")

raw_email = data[0][1]

#Закрываем файлы
f_message.close()

#Удаляем временный файл
os.remove(r'temp')

#Отключаемся от сервера gmail.com
gmail.close()
gmail.logout()