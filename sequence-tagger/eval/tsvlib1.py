#! /usr/bin/env python3

r"""
    This is a small library for reading and interpreting
    the PARSEME TSV format.

    PARSEME TSV files contain these 4 fields:
    * TokenID: an integer or range [in CoNLL-U: ID]
    * Surface: a string [in CoNLL-U: FORM]
    * NoSpace: "nsp" or EMPTY
    * MWECodes: a set of MWE codes (e.g. "3:LVC;2;5:ID") or EMPTY
    Any extra column should be ignored.

    As in CoNLL-U, EMPTY columns are represented by "_".
"""


import collections
import sys

EMPTY = ["_", ""]


class TSVSentence:
    r"""A list of TSVTokens.
        TSVTokens may include ranges and sub-tokens.

        For example, if we have these TSVTokens:
            1   You
            2-3 didn't   -- a range
            2   did      -- a sub-token
            3   not      -- a sub-token
            4   go
        Iterating through `self.words` will yield ["You", "did", "not", "go"].
        You can access the range ["didn't"] through `self.contractions`.
    """
    def __init__(self, filename, lineno_beg, words=None, contractions=None):
        self.filename = filename
        self.lineno_beg = lineno_beg
        self.words = words or []
        self.contractions = contractions or []

    def __str__(self):
        return "TSVSentence({!r}, {!r}, {!r}, {!r})".format(self.filename,
                self.lineno_beg, self.words, self.contractions)

    def append(self, token):
        r"""Add `token` to either `self.words` or `self.contractions`."""
        L = (self.contractions if token.is_contraction() else self.words)
        L.append(token)

    def subtoken_indexes(self):
        r"""Return a set with the index of every sub-word."""
        sub_indexes = set()
        for token in self.contractions:
            sub_indexes.update(token.contraction_range())
        return sub_indexes

    def iter_words_and_ranges(self):
        r"""Yield all tokens, including ranges.
        For example, this function may yield ["You", "didn't", "did", "not", "go"].
        """
        index2contractions = collections.defaultdict(list)
        for c in self.contractions:
            index2contractions[c.contraction_range().start] = c
        for i, token in enumerate(self.words):
            for c in index2contractions[i]:
                yield c
            yield token

    def mwe_infos(self):
        r"""Return a dict {mwe_id: MWEInfo} for all MWEs in this sentence."""
        mwe_infos = {}
        for token_index, token in enumerate(self.words):
            global_last_lineno(self.filename, token.lineno)
            for mwe_id, mwe_categ in token.mwes_id_categ():
                mwe_info = mwe_infos.setdefault(mwe_id, MWEInfo(mwe_categ, []))
                mwe_info.token_indexes.append(token_index)
        return mwe_infos

    def absorb_mwes_from_contraction_ranges(self):
        r"""If a range is part of an MWE, add its subtokens as part of it as well."""
        for c in self.contractions:
            for i_subtoken in c.contraction_range():
                self.words[i_subtoken].mwe_codes.update(c.mwe_codes)




class MWEInfo(collections.namedtuple('MWEInfo', 'category token_indexes')):
    r"""Represents a single MWE in a sentence.
    CAREFUL: token indexes start at 0 (not at 1, as in the TokenID's).

    Arguments:
    @type category: Optional[str]
    @type token_indexes: list[int]
    """
    pass



class TSVToken(collections.namedtuple('Token', 'lineno token_id surface nsp mwe_codes extra')):
    r"""Represents a token in the TSV file.

    Arguments:
    @type lineno: int
    @type token_id: str
    @type surface: str
    @type nsp: bool
    @type mwe_codes: set[str]
    @type extra: list[str]
    """
    def mwes_id_categ(self):
        r"""For each MWE code in `self.mwe_codes`, yield an (id, categ) pair.
        @rtype Iterable[(int, Optional[str])]
        """
        for mwe_code in sorted(self.mwe_codes):
            yield mwe_code_to_id_categ(mwe_code)

    def is_contraction(self):
        r"""Return True iff this token represents a range of tokens.
        (The following tokens in the TSVSentence will contain its elements).
        """
        return "-" in self.token_id

    def contraction_range(self):
        r"""Return a pair (beg, end) with the
        0-based indexes of the tokens inside this range.
        Returns None if not self.is_contraction().
        """
        assert self.is_contraction()
        a, b = self.token_id.split("-")
        return range(int(a)-1, int(b))


def mwe_code_to_id_categ(mwe_code):
    r"""mwe_code_to_id_categ(mwe_code) -> (mwe_id, mwe_categ)"""
    split = mwe_code.split(":")
    mwe_id = int(split[0])
    mwe_categ = (split[1] if len(split) > 1 else None)
    return mwe_id, mwe_categ



############################################################


def iter_tsv_sentences(fileobj):
    r"""Yield `TSVSentence` instances for all sentences in the underlying PARSEME TSV file."""
    sentence = None
    for lineno, line in enumerate(fileobj, 1):
        global_last_lineno(fileobj.name, lineno)
        if line.startswith("#"):
            pass  # Skip comments
        elif line.strip():
            if not sentence:
                sentence = TSVSentence(fileobj.name, lineno)
            fields = line.strip().split('\t')
            extra = fields[4:].copy()
            fields.extend([""]*4)  # fill in the optional fields
            token_id = fields[0]
            surface = fields[1]
            nsp = (fields[2] == 'nsp')
            mwe_codes = [] if fields[3] in EMPTY else fields[3].strip().split(";")
            sentence.append(TSVToken(lineno, token_id, surface, nsp, set(mwe_codes), extra))
        else:
            if sentence:
                yield sentence
                sentence = None
    if sentence:
        yield sentence



####################################################################

last_filename = None
last_lineno = 0

def global_last_lineno(filename, lineno):
    # Update global `last_lineno` var
    global last_filename
    global last_lineno
    last_filename = filename
    last_lineno = lineno


_MAX_WARNINGS = 10
_WARNED = collections.defaultdict(int)

def warn(message, warntype="WARNING", **format_args):
    _WARNED[message] += 1
    if _WARNED[message] <= _MAX_WARNINGS:
        position = "{}:{}: ".format(last_filename, last_lineno) if last_filename else ""
        msg_list = message.format(**format_args).split("\n")
        if _WARNED[message] == _MAX_WARNINGS:
            msg_list.append("(Skipping following warnings of this type)")
        for i, msg in enumerate(msg_list):
            warn = warntype if i==0 else "."*len(warntype)
            print(position, warn, ": ", msg, sep="", file=sys.stderr)

def excepthook(exctype, value, tb):
    global last_lineno
    global last_filename
    if value and last_lineno:
        last_filename = last_filename or "???"
        err_msg = "===> ERROR when reading {} (line {})" \
                .format(last_filename, last_lineno)
        if sys.stderr.isatty():
            err_msg = "\x1b[31m{}\x1b[m".format(err_msg)
        print(err_msg, file=sys.stderr)
    return sys.__excepthook__(exctype, value, tb)


#####################################################################

if __name__ == "__main__":
    sys.excepthook = excepthook
    with open(sys.argv[1]) as f:
        for tsv_sentence in iter_tsv_sentences(f):
            print("TSVSentence:", tsv_sentence)
            print("MWEs:", tsv_sentence.mwe_infos())
