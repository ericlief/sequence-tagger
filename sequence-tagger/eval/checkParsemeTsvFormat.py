#! /usr/bin/env python3

import argparse
import collections
import itertools
import os
import sys
import tsvlib


parser = argparse.ArgumentParser(description="""
        Check if input TSV file is in a valid PARSEME-TSV format.""")
parser.add_argument("tsv_file", type=argparse.FileType('r'),
        help="""The file to be checked""")


class Main(object):
    def __init__(self, args):
        self.args = args
        self.warned = False

    def run(self):
        sys.excepthook = tsvlib.excepthook
        for sentence in tsvlib.iter_tsv_sentences(self.args.tsv_file):
            for i, token in enumerate(sentence.words, 1):
                for mwe_code in token.mwe_codes:
                    try:
                        tsvlib.mwe_code_to_id_categ(mwe_code)
                    except ValueError:
                        self.warn('MWE codes must look like "3:LVC" or "3"\n' \
                                'The MWE code {bad!r} is not well-formed',
                                bad=mwe_code, warntype="ERROR")

                if str(i) != token.token_id:
                    self.warn('Token has rank "{rank}", expected rank "{exp}"',
                            rank=token.token_id, exp=i, warntype="ERROR")

        if not self.warned:
            print('INFO: The file format looks fine!', file=sys.stderr)


    def warn(self, *args, **kwargs):
        self.warned = True
        tsvlib.warn(*args, **kwargs)



#####################################################

if __name__ == "__main__":
    Main(parser.parse_args()).run()
