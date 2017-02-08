import translator as tr
# coding: utf8


def main():


    test = tr.translator('ru')
    x = test.csv_reading('emails.csv')
    k = 1
    for i in x:
        print(k)
        k += 1
        text = test.text_parsing(i)
        tr_text = test.translating(text)
        test.writing(text, tr_text)
        print("wr")
        if k == 100:
            return


if __name__ == '__main__':
    main()
