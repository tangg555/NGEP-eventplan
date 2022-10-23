import pickle
import sys
from nltk.corpus import stopwords
import numpy as np
from gensim.models import KeyedVectors
import spacy
from pathlib import Path

class Coherence(object):
    def __init__(
            self,
            emb_type: str,
            emb_path: str or Path):
        self.stop_words = set(stopwords.words('english'))
        self.nlp = None
        if emb_type == "roberta":
            self.nlp = spacy.load('en_core_web_trf')
        else:
            self.dic_word2vec = self._read_embeddings(emb_type, emb_path)


    def roberta_tokenize(self, text):
        with self.nlp.disable_pipes():
            vectors = np.array([token.vector for token in self.nlp(text)])
        return vectors


    def _read_raw_data(self, path):
        """Return sentences."""
        lines = []
        with open(path) as f:
            for line in f:
                lines.append(line.strip())
        return lines

    def _read_embeddings(self, emb_type, emb_path: Path):
        """Return word embeddings dict."""
        print('[!] Loading word embeddings')
        dic = dict()
        if emb_type == 'glove':
            if emb_path.suffix == ".txt":
                with open(emb_path, 'r') as file1:
                    for line in file1.readlines():
                        row = line.strip().split(' ')
                        emb = np.asarray(row[1:]).astype(np.float32)
                        dic[row[0]] = emb
            elif emb_path.suffix == ".pkl":
                dic = pickle.load(emb_path.open("rb"))
            else:
                raise ValueError(f"_read_embeddings wrong")
        elif emb_type == 'word2vec':
            if emb_path.suffix == ".txt":
                dic = KeyedVectors.load_word2vec_format(str(emb_path), binary=True)
            elif emb_path.suffix == "bin":
                dic = KeyedVectors.load(str(emb_path), mmap='r')
            else:
                raise ValueError(f"_read_embeddings wrong")
        else:
            raise ValueError(f"_read_embeddings wrong")
        #print('[!] Embedding size: ', len(dic))
        assert 'dog' in dic
        print('[!] Load the embedding over')
        return dic

    def _get_vector_of_sentene(self, sentence):
        """Return contains word vector list."""
        remove_stop_word_sentence = [w.lower() for w in sentence.split(' ') if not w.lower() in self.stop_words]
        emb_mean, emb_std = 0, 0.1
        random_emb = np.random.normal(emb_mean, emb_std, (1, len(self.dic_word2vec["love"])))
        vectors = []
        for w in remove_stop_word_sentence:
            if w in self.dic_word2vec:
                vectors.append(self.dic_word2vec[w])
        if len(vectors) == 0:
            vectors = np.asarray(random_emb)
        else:
            vectors = np.asarray(vectors)
        return np.sum(vectors, axis=0)

    def _calc_cosine_sim(self, vectors1, vectors2):
        """Calculate cosine similarity."""
        vectors1 /= np.linalg.norm(vectors1, axis=-1, keepdims=True)
        vectors2 /= np.linalg.norm(vectors2, axis=-1, keepdims=True)
        return np.dot(vectors1, vectors2.T)

    def sentence_coherence_score(
            self,
            response: str,
            context: str) -> float:
        """
        Args:
            response(str): response sentence.
            context: (str): contex tsentence.

        Return:
            float: sentence cosine similarity score

        """
        emb_response = self._get_vector_of_sentene(response)
        emb_context = self._get_vector_of_sentene(context)
        return self._calc_cosine_sim(emb_response, emb_context)