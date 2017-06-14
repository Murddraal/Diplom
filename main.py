"""main moduleof translation programm
"""
import sys
import operator
import getopt
import time
import translator as tr
import re
from my_wordcloud import My_wordcloud as mwc

# coding: utf8

def parsing_arguments():
    # parse commandline arguments
    op = OptionParser()
    op.add_option("--clusterisation",
                action="clustering", type="str")

    op.add_option("--topic_modeling",
                dest="n_components", type="str",
                help="Preprocess documents with latent semantic analysis.")
    op.add_option("--doc_prepocessing",
                action="store_false", type="str", default=True,
                help="Use ordinary k-means algorithm (in batch mode).")
    

    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    return opts




def translating(filename, json_attr="", lang='en'):

    transl = tr.Translator(lang)

    if filename == 'emails.csv':
        csv = tr.csv_reading(filename)

        k = 0
        for i in csv:
            k += 1            
            text = tr.text_parsing_json(i, json_attr)
            text = tr.mail_process(text)
            answ = re.findall('>+', text)
            if len(answ) != 0:
                continue
            if text == "":
                continue
            transl_text = transl.translating(text)

            tr.writing('./latters/transl_latter' + str(k) + '.txt', transl_text)


def detecting(filename, json_attr):

    transl = tr.Translator()

    if filename == 'emails.csv':
        csv = tr.csv_reading(filename)

        for i in csv:
            text = tr.text_parsing_json(i, json_attr)
            transl.lang_detecting(text)

    sorted_langs = sorted(transl.languages.items(),
                          key=operator.itemgetter(1),
                          reverse=True)

    tr.writing_langs("langs.txt", sorted_langs)


def clouding(filename, json_attr):

    if filename == 'emails.csv':
        csv = tr.csv_reading(filename)

        text = ""
        for i in csv:
            text += " " + tr.text_parsing_json(i, json_attr)

        cloud = mwc()
        cloud.gen_cloud('./clouds/cloud_{}.png'.format(filename),
                        './clouds/cloud_{}.txt'.format(filename), text, True)

def translate_then_cloud(filename, json_attr, lang):

    if filename == 'emails.csv':
        transl = tr.Translator(lang)
        csv = tr.csv_reading(filename)

        transl_text = ""
        for i in csv:
            text = tr.text_parsing_json(i, json_attr)
            transl_text += transl.translating(text) + "\n"

        cloud = mwc()
        cloud.gen_cloud('./clouds/cloud_{}.png'.format(filename),
                        './clouds/cloud_{}.txt'.format(filename), text, True)

def main():

   opts = parsing_arguments()

    filename = ""
    start_time = 0
    total_time = 0
    json_attr = ""

    all_opts = [x[0] for x in opts]

    f_opts = [x for x in opts if (x[0] in ["-f", "--file"])]
    if len(f_opts) != 0:
        if f_opts[0][1] != "":
            filename = f_opts[0][1]
        else:
            print("'-f' argument is empty")
            exit(-2)
    else:
        print("there is no '-f' argument")
        exit(-1)

    j_opts = [x for x in opts if (x[0] == "--json")]
    if len(j_opts) != 0:
        a_opts = [x for x in opts if (x[0] in ["-a", "--attr"])]
        try:
            json_attr = a_opts[0][1]
        except TypeError:
            print("Must write json attribute!")
            exit(-1)

    if "--time" in all_opts:
        start_time = time.time()

    for o, a in opts:
        if o == "-t":
            translating(filename, json_attr)
        elif o in ("-T", "--translate"):
            translating(filename, json_attr, a)
        elif o in ("-d", "--detect"):
            detecting(filename, json_attr)
        elif o in ("-c", "--cloud"):
            clouding(filename, json_attr)
        elif o == "t_c":
            translate_then_cloud(filename, json_attr, a)
        elif o in ("-f", "--file", "--time", "--json", "-a", "--attr"):
            continue
        else:
            assert False, "unhandled option"


    if "--time" in all_opts:
        total_time = time.time() - start_time
        print("Time: {}".format(total_time))


if __name__ == '__main__':
    main()
