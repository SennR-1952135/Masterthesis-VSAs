import gensim
from tqdm import tqdm
import nltk
nltk.download('punkt')

# Look into lemmatizing and stemming
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()

import sys
sys.path.append('..')
from shared_code.classes import Dictionary


class Document:
    def __init__(self, title='', text=['']):
        self.dictionary = Dictionary()
        self.set_title(title)
        self.set_text(text)
    
    def set_title(self, title):
        self.title = title

    def set_text(self, text):
        self.text = text
        self.dictionary = Dictionary()
        for word in set(text):
            self.dictionary.add_word(word)
    
class WikiTextDataset:
    """Tokenization and vocabulary building for text corpus."""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.documents: list[Document] = []
        self.dictionary = Dictionary()
        self.corpus = []
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

        for batch in tqdm(self.data_loader):
            for text_fragment in batch:
                # if text fragment contains exactly 2 = signs, it is a title
                if text_fragment.count("=") == 2:
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
                        self.dictionary.add_word(word)
    
    def _add_document(self, title, text):
        doc = Document(title, text)
        self.documents.append(doc)

    def _process_text_fragment(self, text_fragment):
        # skip subheadings
        if "==" in text_fragment:
            return
        # skip empty lines
        if text_fragment.strip() == "":
            return
        # remove <unk> tokens
        text_fragment = text_fragment.replace("<unk>", "")

        # sentence_tokenized = [word_tokenize(sentence) for sentence in sent_tokenize(text_fragment)]
        # sentence_tokenized = [[word.lower() for word in sentence] for sentence in sentence_tokenized]
        sentence_tokenized = gensim.utils.simple_preprocess(text_fragment, deacc=True)
        sentence_tokenized = gensim.parsing.preprocessing.remove_stopword_tokens(sentence_tokenized)
        return sentence_tokenized

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, token):
        return self.documents[token]

    def __iter__(self):
        return iter(self.documents)
    
    def get_word_frequency(self, word, document_frequency=False):
        if document_frequency:
            return sum([word in doc.dictionary.idx2word for doc in self.documents])
        return sum([doc.text.count(word) for doc in self.documents])
