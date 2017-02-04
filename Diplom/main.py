import translator as tr

def main():
    test = tr.translator()
    x = test.csv_reading('emails.csv')
    # for i in x:
    #     text = test.text_parsing(i[0])
    #     tr_text = test.translating(text)
    #     print('Translate:\n' + tr_text)

    test.writing('', '')
        

if __name__ == '__main__':
	main()