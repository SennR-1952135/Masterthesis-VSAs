import gensim
from tqdm import tqdm
import pyarrow.parquet as pq
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')

# Look into lemmatizing and stemming
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()

import sys
sys.path.append('..')
from shared_code.classes import Vocabulary


class Document:
    def __init__(self, title='', text=['']):
        self.vocabulary = Vocabulary()
        self.set_title(title)
        self.set_text(text)
    
    def set_title(self, title):
        self.title = title

    def set_text(self, text):
        self.text = text
        self.vocabulary = Vocabulary()
        for word in set(text):
            self.vocabulary.add_word(word)

def ParquetFileIterator(file_name, batch_size=100):
    parquet_file = pq.ParquetFile(file_name)
    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        for d in record_batch.to_pylist():
            yield d
    
class WikiTextDataset:
    """Tokenization and vocabulary building for text corpus."""

    def __init__(self, input_file, out_dir):
        self.data_file = input_file
        self.out_dir = out_dir
        self.documents: list[Document] = []
        self.vocabulary = Vocabulary()
        self.max_documents = 3000

        self.process_data()

    def process_data(self):
        """
          Tokenize the lines, remove the titles, and make it lowercase,
          return lines list.
          list[list[word]]
        """
        current_text_tokens = []
        current_title = ''

        for pqf in ParquetFileIterator(self.data_file):
                text_fragment = pqf['text']
                # if text fragment contains exactly 2 = signs, it is a title
                text_fragment = text_fragment.strip()
                if text_fragment.count("=") == 2 and text_fragment.startswith("=") and text_fragment.endswith("="):
                    if current_text_tokens:
                      self._add_document(current_title, current_text_tokens)
                    
                    current_title = text_fragment.replace("=", "").strip()
                    current_text_tokens = []
                    if len(self.documents) == self.max_documents:
                        return
                    continue
                
                sentence_tokenized = self._process_text_fragment(text_fragment)
                if sentence_tokenized:
                    current_text_tokens.extend(sentence_tokenized)
                    for word in sentence_tokenized:
                        self.vocabulary.add_word(word)
    
    def _add_document(self, title, text):
        doc = Document(title, text)
        self.documents.append(doc)


    def process_title(self, title):
        return title

    def _process_text_fragment(self, text_fragment):
        # skip subheadings
        if "==" in text_fragment:
            return
        # skip empty lines
        if text_fragment.strip() == "":
            return
        # remove <unk> tokens
        text_fragment = text_fragment.replace("<unk>", "")
        # text_fragment = re.sub(r'\d+', '<num>', text_fragment)

        sentence_tokenized = word_tokenize(text_fragment)
        sentence_tokenized = [word.lower() for word in sentence_tokenized if not (word in stopwords.words() or word in string.punctuation + '``' + "''")]
        sentence_tokenized = [word if word.isalpha() else '<num>' for word in sentence_tokenized]
        
        # sentence_tokenized = [word_tokenize(sentence) for sentence in sent_tokenize(text_fragment)]
        # sentence_tokenized = [[word.lower() for word in sentence] for sentence in sentence_tokenized]
        # sentence_tokenized = gensim.utils.simple_preprocess(text_fragment, deacc=True)
        # sentence_tokenized = gensim.parsing.preprocessing.remove_stopword_tokens(sentence_tokenized)
        return sentence_tokenized

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, token):
        return self.documents[token]

    def __iter__(self):
        return iter(self.documents)
    
    def get_word_frequency(self, word, document_frequency=False):
        if document_frequency:
            return sum([word in doc.vocabulary.idx2word for doc in self.documents])
        return sum([doc.text.count(word) for doc in self.documents])
    
# class WikiTextDataset:
#     """Tokenization and vocabulary building for text corpus."""

#     def __init__(self, data_loader):
#         self.data_loader = data_loader
#         self.documents: list[Document] = []
#         self.dictionary = Dictionary()
#         self.corpus = []
#         self.max_documents = 3000

#         self.process_data()

#     def process_data(self):
#         """
#           Tokenize the lines, remove the titles, and make it lowercase,
#           return lines list.
#           list[list[word]]
#         """
#         current_text_tokens = []
#         current_title = ''

#         for batch in tqdm(self.data_loader):
#             for text_fragment in batch:
#                 # if text fragment contains exactly 2 = signs, it is a title
#                 if text_fragment.count("=") == 2:
#                     if current_text_tokens:
#                       self._add_document(current_title, current_text_tokens)
                    
#                     current_title = text_fragment.replace("=", "").strip()
#                     current_text_tokens = []
#                     if len(self.documents) == self.max_documents:
#                         return
#                     continue
                
#                 sentence_tokenized = self._process_text_fragment(text_fragment)
#                 if sentence_tokenized:
#                     current_text_tokens.extend(sentence_tokenized)
#                     for word in sentence_tokenized:
#                         self.dictionary.add_word(word)
    
#     def _add_document(self, title, text):
#         doc = Document(title, text)
#         self.documents.append(doc)

#     def _process_text_fragment(self, text_fragment):
#         # skip subheadings
#         if "==" in text_fragment:
#             return
#         # skip empty lines
#         if text_fragment.strip() == "":
#             return
#         # remove <unk> tokens
#         text_fragment = text_fragment.replace("<unk>", "")

#         # sentence_tokenized = [word_tokenize(sentence) for sentence in sent_tokenize(text_fragment)]
#         # sentence_tokenized = [[word.lower() for word in sentence] for sentence in sentence_tokenized]
#         sentence_tokenized = gensim.utils.simple_preprocess(text_fragment, deacc=True)
#         sentence_tokenized = gensim.parsing.preprocessing.remove_stopword_tokens(sentence_tokenized)
#         return sentence_tokenized

#     def __len__(self):
#         return len(self.documents)

#     def __getitem__(self, token):
#         return self.documents[token]

#     def __iter__(self):
#         return iter(self.documents)
    
#     def get_word_frequency(self, word, document_frequency=False):
#         if document_frequency:
#             return sum([word in doc.dictionary.idx2word for doc in self.documents])
#         return sum([doc.text.count(word) for doc in self.documents])
