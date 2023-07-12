import uuid
from pathlib import Path

import spacy
from tqdm import tqdm
from tqdm.auto import tqdm

from ..logger import logger

import re
from collections import Counter


class Corpus:
    """ """
    REGISTER_WEB = "web"
    REGISTER_NEWS = "news"
    LANGUAGE_FR = "fr"
    LANGUAGE_EN = "en"

    def __init__(self, register, language, input_path):
        """Initialise a corpus object.

        :param register: Register of data in input.
        :type register: str
        :param language: Language of the data in input.
        :type language: str
        :param input_path: Input path for the data.
        :type input_path: str
        """
        self.language = language
        self.register = register
        self.input_path = input_path


class VRTMaker:
    """ """
    def _get_model(self):
        """Load a SpaCy model.


        :returns: A spacy.Language object with the loaded pipeline.

        :rtype: spacy.Language

        """
        if self.corpus.language == "fr":
            return spacy.load("fr_core_news_lg")
        elif self.corpus.language == "en":
            return spacy.load("en_core_web_lg")

    def _create_output(self, output_path):
        """Create the output file for the processed data.

        :param output_path: Path in which to output the data.
        :type output_path: str
        :returns: The filepath in output.
        :rtype: str

        """
        op = Path(output_path)
        op.mkdir(exist_ok=True)
        corpus_output = op / f"{Path(self.corpus.input_path).stem}.vrt"  # Output path to write the corpus file
        corpus_output.touch()
        return corpus_output

    def __init__(self, corpus: Corpus, output_path, n_jobs=1):
        """Initialize a VRTMaker object to build VRT preprocessed corpus files.

        :param corpus: The corpus object to preprocess.
        :type corpus: Corpus
        :param output_path: The output filepath to write VRT file.
        :type output_path: str
        :param n_jobs: Number of jobs for preprocessing steps, defaults to 1
        :type n_jobs: int, optional
        """
        self.corpus = corpus
        self.corpus_output = self._create_output(output_path)
        self.model = self._get_model()
        self._with_ner_merging()
        self.n_jobs = n_jobs

    def _open(self):
        """Open the output file.


        :returns: The output buffer.

        :rtype: file

        """
        fichier = self.corpus_output.open("w")
        return fichier

    def _with_ner_merging(self):
        """Merge named entities."""
        self.model.add_pipe("merge_entities", after="ner")  # Merge Named-Entities

    def do_txt_to_vrt(self):
        """Build VRT format file and write to output filepath."""
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

def extract_text(corpus_path, exceptions_path=None, lemmatize=True, stop_words=False, lower_words=True, number=False, punct=False, exclude_pos=[], en="chunking", min_freq=50, alpha=True, exclude_en=[], min_length_word=3):
    """Extracts the text from a VRT corpus file.

    :param corpus_path: str
    :param lemmatize: bool (Default value = True)
    :param stop_words: bool (Default value = False)
    :param lower: bool
    :param number: bool (Default value = False)
    :param punct: bool (Default value = False)
    :param exclude_pos: list (Default value = [])
    :param en: str ("chunking", "tagging", "deleting") (Default value = "chunking")
    :param min_freq: int (Default value = 50)
    :param alpha: bool (Default value = True)
    :param exclude_en: list (Default value = [])
    :param lower_words:  (Default value = True)
    :param min_length_word:  (Default value = 3)
    :returns: text (list(list(str))): A list of sentences containing words

    """
    corpus_file = open_corpus(corpus_path)
    text = corpus_file.read().splitlines()
    out = []
    pattern = re.compile(r"<text[^<>]*\"\>{1}")
    stop_words, number, punct, alpha = str(stop_words), str(number), str(punct), str(alpha)
    sentence = []
    
    if en != "chunking" and en != "tagging" and en != "deleting" :
        logger.info(f"No correct option for en was provided: {en} is not valid. en option was thus set to chunking")
        en = "chunking"
        
    if exceptions_path != None :
        exceptions_file = open_corpus(exceptions_path)
        exceptions = exceptions_file.read().splitlines()
        if lower_words:
            exceptions = [w.lower() for w in exceptions]
    else : 
        exceptions = []
        
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
                if lower_words:
                    token_ = token.lower()
                    lemma_ = lemma.lower()
                else:
                    token_ = token
                    lemma_ = lemma
                if not lemmatize:
                    lemma_ = token_
                if token_ in exceptions : 
                    sentence.append(token_)
                else :
                    if stop_words == is_stop and is_punct == punct and is_digit == number and like_num == number and not pos in exclude_pos and not ent_type in exclude_en and (alpha == is_alpha or ent_type != "None"):
                        if exclude_en and ent_iob != "None":
                            pass
                        else:
                            if ent_type != "None" and len(lemma_) > 1:
                                if en == "chunking" :
                                    sentence.append(token_)
                                elif en == "tagging" :
                                    sentence.append(ent_type)  
                                elif en == "deleting" :
                                    pass
                            elif len(lemma) > min_length_word:
                                sentence.append(lemma_)
                    else:
                        pass
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
    """

    :param corpus_path: 

    """
    if isinstance(corpus_path, str):
        corpus_file = Path(corpus_path).open("r")
    elif isinstance(corpus_path, Path):
        corpus_file = corpus_path.open("r")
    else:
        raise TypeError(
            f"Path to corpus : corpus_path must be of type int or pathlib.Path. Provided corpus_path type is {corpus_path.type}")
    return corpus_file
