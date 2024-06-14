"""Tests for `sinr_embeddings` package.
Testing preprocessing of a textual corpus.
"""

import pytest
import unittest

import sinr.text.preprocess as ppcs
import urllib.request
import os
from sklearn.datasets import fetch_20newsgroups



class TestSinr_embeddings(unittest.TestCase):
    """Tests for `graph_embeddings` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        
        txt_path = './ppcs_test.txt'
        vrt_path = './ppcs_test.vrt'
        except_path = './except.txt'
        txt_empty_docs_path = './ppcs_test_empty_docs.txt'
        vrt_empty_docs_path = './ppcs_test_empty_docs.vrt'
        doc_separator = '##D'
        s0 = doc_separator + " At 10 a.m, in the heart of New York City, John Smith walked briskly towards the Empire State Building. "
        text = ( s0 +
                "As he passed by Starbucks, he grabbed a quick coffee. "
                "He was on his way to meet Sarah Johnson, the CEO of GlobalTech, for an important business meeting. "
                + doc_separator +
                "Sarah Johnson, known for her sharp business acumen, was already waiting at the lobby of the iconic skyscraper. "
                "As John entered the building, he couldn't help but feel a sense of nervous excitement. "
                + doc_separator +
                "The meeting room was located on the 60th floor, overlooking the bustling city below.\n"
                "After a firm handshake and brief introductions, John and Sarah delved into discussions about potential collaborations. "
                "The conversation flowed smoothly, with both parties expressing enthusiasm for the possibilities ahead. "
                "By the end of the meeting, they had outlined a preliminary agreement and scheduled further negotiations for the following week. "
                + doc_separator +
                "With a renewed sense of optimism, John left the Empire State Building, knowing that his encounter with Sarah Johnson could mark a"
                "significant turning point for Tech Innovations Inc.")
        with open(txt_path, 'w+') as file:
            file.write(text)
        file.close()
        with open(txt_empty_docs_path, 'w+') as file:
            file.write(doc_separator + ' ' + doc_separator + ' ' + doc_separator + ' ' + doc_separator + ' ')
        file.close()
        with open(except_path, 'w+') as file:
            file.write('at\nin')
        file.close()
        self.txt_path = txt_path
        self.vrt_path = vrt_path
        self.except_path = except_path
        self.txt_empty_docs_path = './ppcs_test_empty_docs.txt'
        self.vrt_empty_docs_path = './ppcs_test_empty_docs.vrt'
        self.n_doc = 4
        self.n_sent = 10
        self.doc_separator = doc_separator
        self.s0 = s0
        
    def tearDown(self):
        """Tear down test fixtures, if any."""
        os.remove(self.txt_path)
        os.remove(self.except_path)
        os.remove(self.txt_empty_docs_path)
        if os.path.isfile(self.vrt_path):
            os.remove(self.vrt_path)
        else:
            os.remove(self.vrt_empty_docs_path)
    
    def test_doc_separator(self):
        """Testing if the preprocessed datas have the right number of documents"""
        vrt_maker = ppcs.VRTMaker(ppcs.Corpus(ppcs.Corpus.REGISTER_WEB,
                                  ppcs.Corpus.LANGUAGE_EN,
                                  self.txt_path),
                                  ".", n_jobs=8, spacy_size='sm')
        vrt_maker.do_txt_to_vrt(separator=self.doc_separator)
        docs = ppcs.extract_text(self.vrt_path, lemmatize=True, min_freq=1)
        self.assertTrue(len(docs) == self.n_doc)
        
    def test_sentence_separator(self):
        """Testing if the preprocessed datas have the right number of sentences"""
        vrt_maker = ppcs.VRTMaker(ppcs.Corpus(ppcs.Corpus.REGISTER_WEB,
                                  ppcs.Corpus.LANGUAGE_EN,
                                  self.txt_path),
                                  ".", n_jobs=8, spacy_size='sm')
        vrt_maker.do_txt_to_vrt()
        sentences = ppcs.extract_text(self.vrt_path, lemmatize=True, min_freq=1)
        self.assertTrue(len(sentences) == self.n_sent)
    
    def test_preprocessed(self):
        """Testing the preprocessing by default on the first sentence"""
        vrt_maker = ppcs.VRTMaker(ppcs.Corpus(ppcs.Corpus.REGISTER_WEB,
                                  ppcs.Corpus.LANGUAGE_EN,
                                  self.txt_path),
                                  ".", n_jobs=8, spacy_size='sm')
        vrt_maker.do_txt_to_vrt()
        sentences = ppcs.extract_text(self.vrt_path, min_freq=1)
        
        doc = vrt_maker.model(self.s0)
        s = sentences[0]
        ok = list()
        ind = -1
        for i,token in enumerate(doc):
            if not token.is_punct and not token.is_stop and not token.is_digit and not token.like_num:
                if token.ent_type_ == '':
                    if len(token.lemma_) > 3:
                        ind += 1
                        ok.append(token.lemma_.lower() == s[ind])
                else:
                    if len(token.text) > 3:
                        ind += 1
                        if ' ' in token.text:
                            ok.append(token.text.replace(' ', '_').lower() == s[ind])
                        else:
                            ok.append(token.text.lower() == s[ind])
                            
        self.assertTrue(False not in ok)
        
    def test_preprocessing_empty_docs(self):
        """Testing min_length_doc = -1 : documents of all sizes are kept"""
        vrt_maker = ppcs.VRTMaker(ppcs.Corpus(ppcs.Corpus.REGISTER_WEB,
                                  ppcs.Corpus.LANGUAGE_EN,
                                  self.txt_empty_docs_path),
                                  ".", n_jobs=8, spacy_size='sm')
        vrt_maker.do_txt_to_vrt(separator=self.doc_separator)
        docs = ppcs.extract_text(self.vrt_empty_docs_path, min_freq=1, min_length_doc=-1)
        self.assertTrue(len(docs) == self.n_doc)
    
    def test_exceptions_list(self):
        """Testing the preprocessing with an exceptions list"""
        vrt_maker = ppcs.VRTMaker(ppcs.Corpus(ppcs.Corpus.REGISTER_WEB,
                                  ppcs.Corpus.LANGUAGE_EN,
                                  self.txt_path),
                                  ".", n_jobs=8, spacy_size='sm')
        vrt_maker.do_txt_to_vrt()
        sentences = ppcs.extract_text(self.vrt_path, min_freq=1, exceptions_path=self.except_path)
        
        self.assertTrue('at' in sentences[0] and 'in' in sentences[0])
            
        
if __name__ == '__main__':
    unittest.main()