# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the Stanford Tokenizer
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Steven Xu <xxu@student.unimelb.edu.au>
#
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from nltk.tokenize.api import TokenizerI


class BaseTokenizer(TokenizerI):
    r"""
    Interface to the Stanford Tokenizer

    >>> s = "The colour of the wall is blue."
    >>> StanfordTokenizer(options={"americanize": True}).tokenize(s)
    ['The', 'color', 'of', 'the', 'wall', 'is', 'blue', '.']
    """
    def __init__(self, client, options=None):
        self._client = client
        options = {} if options is None else options
        self._options = ','.join('{0}={1}'.format(key, val) for key, val in options.items())

    def tokenize(self, s):
        """
        Use stanford tokenizer's PTBTokenizer to tokenize multiple sentences.
        """
        return self._execute(s)

    def _execute(self, annotators, s):
        properties = {'annotators': ",".join(annotators),
                      'inputFormat': 'text',
                      'outputFormat': 'serialized',
                      'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'}
        if self._options:
            properties['tokenize.options'] = self._options
        return self._client.annotate(s, properties=properties)


class WordTokenizer(BaseTokenizer):
    def __init__(self, client, options=None):
        super(WordTokenizer, self).__init__(client, options)

    def tokenize(self, s):
        doc = self._execute(['tokenize'], s)
        for t in doc.sentencelessToken:
            yield t.word


class SentTokenizer(BaseTokenizer):
    def __init__(self, client, options=None):
        super(SentTokenizer, self).__init__(client, options)

    def tokenize(self, s):
        doc = self._execute(['ssplit'], s)
        # for s in doc.sentence:
        #     reconstruct = ''
        #     for t in s.token:
        #         reconstruct += t.before + t.word
        #     yield reconstruct
        for s in doc.sentence:
            yield [t.word for t in s.token]
