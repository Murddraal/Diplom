import translator as tr
import operator
import time
# coding: utf8


def main():


    test = tr.translator('ru')
    x = test.csv_reading('emails.csv')
    k = 1
    start_time = time.time()
    for i in x:
        #print(k)
        k += 1
        text = test.text_parsing(i)
        # tr_text = test.translating(text)
        # test.writing(text, tr_text)
        # print("wr")
        langs = test.lang_detecting(text, 1)
    total_time = time.time() - start_time

    sorted_langs = sorted(langs.items(), key=operator.itemgetter(1), reverse=False)
    
    sorted_langs.reverse()
    with open('langs_ya.txt', 'w') as f:
        f.write('Time: ' + str(total_time) + '\n\n')
        for i in sorted_langs:
            f.write(str(i[0]) + '-' + str(i[1]) + '\n')

if __name__ == '__main__':
    main()
