#!/usr/bin/env python3

from os import chdir, listdir
from os.path import dirname, join, realpath
from pathlib import Path
import re


def compile():
    """Compile HTML files"""

    # operate in this file's direcctory
    chdir(dirname(realpath(__file__)))

    # compile each file
    folder = 'html'
    for fname in listdir(folder):

        # only do html files
        if fname.rsplit('.')[-1] != 'html':
            continue

        # load file
        contents = open(join(folder, fname), 'r').read()

        # replace directives
        directives = re.findall(r'<!-- html/partial/.*.html -->', contents)
        for directive in directives:
            partial_fname = directive[5:-4]
            partial_contents = open(partial_fname, 'r').read()
            contents = contents.replace(directive, partial_contents)

        # minify
        contents = re.sub(r'<!-- .* -->', '', contents)
        contents = re.sub(r'[ ]{2,}', ' ', contents)
        contents = re.sub(r'^ ', '', contents, flags=re.MULTILINE)
        contents = re.sub(r'\n{2,}', '\n', contents, flags=re.MULTILINE)

        # create output folder (if missing)
        Path("site").mkdir(exist_ok=True)

        # write compiled html file
        ofname = 'site/' + fname.rsplit('.')[0]
        print(contents, file=open(ofname, 'w'))


# command-line interface
if __name__ == '__main__':
    compile()
