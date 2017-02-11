import translator as tr
# coding: utf8


def main():


    test = tr.translator('ru')
    x = test.csv_reading('emails.csv')
    k = 1
    for i in x:
        #print(k)
        k += 1
        text = test.text_parsing(i)
        # tr_text = test.translating(text)
        # test.writing(text, tr_text)
        # print("wr")
        langs = test.lang_detecting(text, 0)
        if k==120:
            break
    print(langs)
    print(sorted(langs, key=langs.__getitem__))


if __name__ == '__main__':
    main()
