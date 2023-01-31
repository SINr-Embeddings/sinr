import uuid
from pathlib import Path

import spacy
from tqdm import tqdm
from tqdm.auto import tqdm

from ..logger import logger

import re
from collections import Counter


class Corpus:
    REGISTER_WEB = "web"
    REGISTER_NEWS = "news"
    LANGUAGE_FR = "fr"
    LANGUAGE_EN = "en"

    def __init__(self, register, language, input_path):
        self.language = language
        self.register = register
        self.input_path = input_path


class VRTMaker:
    def _get_model(self):
        if self.corpus.language == "fr":
            return spacy.load("fr_core_news_lg")
        elif self.corpus.language == "en":
            return spacy.load("en_core_web_lg")

    def _create_output(self, output_path):
        op = Path(output_path)
        op.mkdir(exist_ok=True)
        corpus_output = op / f"{Path(self.corpus.input_path).stem}.vrt"  # Output path to write the corpus file
        corpus_output.touch()
        return corpus_output

    def __init__(self, corpus: Corpus, output_path, n_jobs=1):
        self.corpus = corpus
        self.corpus_output = self._create_output(output_path)
        self.model = self._get_model()
        self._with_ner_merging()
        self.n_jobs = n_jobs

    def _open(self):
        fichier = self.corpus_output.open("w")
        return fichier

    def _with_ner_merging(self):
        self.model.add_pipe("merge_entities", after="ner")  # Merge Named-Entities

    def do_txt_to_vrt(self):
        corpus_opened = self._open()
        id_corpus = str(uuid.uuid4())  # Generate a random corpus id
        corpus_opened.write(
            f'<text id="{id_corpus}" filename="{Path(self.corpus.input_path).as_posix()}" register="{self.corpus.register}" language="{self.corpus.language}">\n')  # Write corpus identifier
        input_file = Path(self.corpus.input_path).open("r")
        input_txt = input_file.read().splitlines()  # Read INPUT_FILE
        logger.info(str(len(input_txt)) + "lines to preprocess")
        input_file.close()
        for doc in tqdm(self.model.pipe(input_txt, n_process=self.n_jobs), total=len(input_txt)):
            for sent in doc.sents:  # Sentence border detection
                corpus_opened.write("<s>\n")  # Write a sentence start
                for token in sent:  # For each token
                    if token.ent_type_ == '':  # If current token is not a Named-Entity
                        ent_type = None
                        text = token.text
                        lemma = token.lemma_
                    else:
                        ent_type = token.ent_type_
                        if ' ' in token.text:  # Entities are merged with a space by default
                            text = token.text.replace(' ', '_')  # We want to merge named entities with a _
                            lemma = text
                        else:
                            text = token.text
                            lemma = text
                    content = "\t".join([text,
                                         lemma,
                                         token.pos_,
                                         token.ent_iob_,
                                         str(ent_type),
                                         str(token.is_punct),
                                         str(token.is_stop),
                                         str(token.is_alpha),
                                         str(token.is_digit),
                                         str(token.like_num)])
                    corpus_opened.write(f'{content}\n')  # Write the token info
                corpus_opened.write("</s>\n")
        corpus_opened.close()
        logger.info(f"VRT-style file written in {self.corpus_output.absolute()}")


def extract_text(corpus_path, lemmatize=True, stop_words=False, lower_words=True, number=False, punct=False,
                 exclude_pos=[],
                 en=True, min_freq=50, alpha=True, exclude_en=[], min_length_word=3):
    '''corpus_path
    Extracts the text from a VRT corpus file.


    Parameters:
    corpus_path (str|pathlib.Path): Path to the corpus file.
    lemmatize (bool): Return lemmatized text (default: True).
    stop_words (bool): Keep stop-words (default: False).
    lower (bool): Put the text in lowercase (default: True).
    number (bool): Keep the numbers (default: False).
    punct (bool): Keep the punctuation (default: False).
    exclude_pos (list): List of part-of speech (from spacy) to exclude) (default: []).
    en (bool): Keep named entities (default:True)
    min_freq (int): Minimum number of occurrences to keep a token (default: 10).
    alpha (bool): Keep alphanumeric characters (default: False).
    exclude_en (list): List of named-entities types to exclude (default: []).


    Return:
    text (list(list(str))): A list of sentences containing words
    '''
    corpus_file = open_corpus(corpus_path)
    text = corpus_file.read().splitlines()
    out = []
    pattern = re.compile(r"<text[^<>]*\"\>{1}")
    stop_words, number, punct, alpha = str(stop_words), str(number), str(punct), str(alpha)
    sentence = []

    for line in tqdm(text, total=len(text)):
        if line.startswith("<s>"):
            sentence = []
        elif line.startswith("</s>"):
            if len(sentence) > 2:
                out.append(sentence)
        elif len(pattern.findall(line)) > 0:
            pass
        else:
            listline = line.split("\t")
            if len(listline) == 10:
                for i in listline:
                    if bool(re.match('^\t\t', str(i))):
                        continue
                token, lemma, pos, ent_iob, ent_type, is_punct, is_stop, is_alpha, is_digit, like_num = line.split("\t")
                if lemmatize:
                    if stop_words == is_stop and is_punct == punct and is_digit == number and like_num == number and not pos in exclude_pos and not ent_type in exclude_en and (
                            alpha == is_alpha or ent_type != "None"):
                        if exclude_en and ent_iob != "None":
                            pass
                        else:
                            if lower_words:
                                if ent_type != "None" and len(lemma) > 1:
                                    sentence.append(token)  # sentence.append(lemma.lower())
                                    # print(lemma)
                                elif len(lemma) > min_length_word:
                                    sentence.append(lemma.lower())
                            else:
                                if ent_type != "None":
                                    sentence.append(token)
                                elif len(lemma) > min_length_word:
                                    sentence.append(lemma)
                    else:
                        pass
                else:
                    if stop_words == is_stop and is_punct == punct and is_digit == number and alpha == is_alpha and like_num == number and not pos in exclude_pos and not ent_type in exclude_en:
                        if exclude_en and ent_iob != "None":
                            pass
                        else:
                            if lower_words:
                                if ent_type != "None" and len(token) > 1:
                                    sentence.append(token)  # (token)
                                elif len(token) > min_length_word:
                                    sentence.append(token.lower())
                            else:
                                if ent_type != "None":
                                    sentence.append(token)  # (token)
                                elif len(lemma) > min_length_word:
                                    sentence.append(token)
            else:
                continue
    if min_freq > 1:
        counts = Counter([word for sent in out for word in sent])
        accepted_tokens = {word for word, count in counts.items() if count >= min_freq}
        out = [[word for word in sent if word in accepted_tokens] for sent in out]
        # line = corpus.readline().rstrip()
        # x+=1
    return out


def open_corpus(corpus_path):
    if isinstance(corpus_path, str):
        corpus_file = Path(corpus_path).open("r")
    elif isinstance(corpus_path, Path):
        corpus_file = corpus_path.open("r")
    else:
        raise TypeError(
            f"Path to corpus : corpus_path must be of type int or pathlib.Path. Provided corpus_path type is {corpus_path.type}")
    return corpus_file
