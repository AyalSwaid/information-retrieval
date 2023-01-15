import os
from pathlib import Path
import pickle
# from nltk.corpus import stopwords
import re
from collections import Counter, OrderedDict, defaultdict
import hashlib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from inverted_index_gcp import InvertedIndex

def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124
def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS

class Index:
    """
    This is the main index of the project, so it is the only class that communicate directly with the inverted indices.
    this class is only used by Engine class to help searching queries.
    contains three InvertedIndex objects: body_index, title_index, anchor_index.
    self.__init__ inits empty indices objects, and self.read_indices will read the indices contents from disk.
    """
    def __init__(self):
        self.all_pls_titles = defaultdict(list)
        # self.all_pls_body = defaultdict(list)
        self.all_pls = {'title_index':defaultdict(list),
                        'body_index': defaultdict(list)}

        # init indeices
        self.titles_index = InvertedIndex(dir_name='title_index/')
        self.body_index = InvertedIndex(dir_name = 'body_index/')
        self.anchor_index = InvertedIndex(dir_name = 'anchor_index/')


    def read_indices(self):
        """
        read inverted indices from relevant directory and store them in self.titles_index, self.body_index, self.anchor_index
        this fnction can be used only after indidces are already built.
        :return: None
        """
        print('Reading indices...')
        self.titles_index = InvertedIndex.read_index('title_index', 'title_index')
        self.body_index = InvertedIndex.read_index('body_index', 'body_index')
        self.anchor_index = InvertedIndex.read_index('anchor_index', 'anchor_index')
        
        self.titles_index.dir_name = 'title_index/'
        self.body_index.dir_name = 'body_index/'
        self.anchor_index.dir_name = 'anchor_index/'
        
    def load_file(self):
        """
        read local pkl file to test the model on my pc
        :return:
        """
        pkl_file = "part15_preprocessed.pkl"

        try:
            if os.environ["assignment_2_data"] is not None:
                pkl_file = Path(os.environ["assignment_2_data"])
        except:
            Exception("Problem with one of the variables")

        assert os.path.exists(pkl_file), 'You must upload this file.'
        with open(pkl_file, 'rb') as f:
            pages = pickle.load(f)
        return pages

    def word_count(self, text, id):
        ''' Count the frequency of each word in `text` (tf) that is not included in
        `all_stopwords` and return entries that will go into our posting lists.
        Parameters:
        -----------
            text: str
            Text of one document
            id: int
            Document id
        Returns:
        --------
            List of tuples
            A list of (token, (doc_id, tf)) pairs
            for example: [("Anarchism", (12, 5)), ...]
        '''
        tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
        counts = {}
        for t in tokens:
            if t not in all_stopwords:
                counts[t] = counts.get(t, 0)+1
        return [(w, (id,f)) for (w,f) in counts.items()], tokens

    def union_pls(self, pls, index_type, all_pls):
        for w, lst in pls:
            all_pls[index_type][w].append(lst)
            # self.all_pls_titles[w].append(lst)


    def tokenize(self, txt):
        """
        tokenize txt and delete stopwords
        :param txt: 
        :return: 
        """""
        return [token.group() for token in RE_WORD.finditer(txt.lower()) if token.group() not in all_stopwords]

# some tests
if __name__ == "__main__":
    my_index = Index()
    my_index.build_indices()
    print("before read locs:", len(my_index.titles_index.posting_locs))
    my_index.read_indices()
    print("after read locs:", len(my_index.titles_index.posting_locs))
    print(len(my_index.body_index.posting_locs))
    # print(my_index.titles_index.posting_locs)
