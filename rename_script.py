# import os, sys

# str_train = 'train'
# str_test = 'test'
# main_dir = './20news-bydate/20news-bydate-{}/comp.graphics/'.format(typ)
# DIR = './20news-bydate/20news-bydate-{}/comp.graphics/'.format(typ)
# files = os.listdir(DIR) # use the full path to the folder as an arugument to the script
# files.sort()
# n=1
# for f in files:
#     os.rename(DIR+f, DIR+'{}_comp_graphics_{}'.format(typ, n))
#     n += 1


import os, sys, re

def rename(typ, folder):
    DIR = './20news-bydate/20news-bydate-{}/{}/'.format(typ, folder)
    files = os.listdir(DIR) # use the full path to the folder as an arugument to the script

    filename = re.sub(r'\.', '_', folder)

    n = 1
    for f in files:
        os.rename(DIR+f, DIR+'{}_{}_{}'.format(typ, filename, n))
        n += 1

str_train = 'train'
str_test = 'test'
main_dir = './20news-bydate/20news-bydate-test/'
folders = os.listdir(main_dir)

for folder in folders:

    rename(str_train, folder)
    rename(str_test, folder)