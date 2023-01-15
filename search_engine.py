from build_index import Index
from collections import Counter, OrderedDict, defaultdict
from time import time
import pandas as pd
import numpy as np
import pickle


class Engine:
    """
    This is the main Engine in this prject, this class takes a query directly from the flask app and do the search
    using Index class(which includes the inverted indices)

    """
    def __init__(self):
        """
        init indices (body, title, anchor), and read pageRank and pageViews data, and read id2title pickle which contains the title
        for each doc id. Then init title_words, body_words, anchor_words which are the term_total dict of each
        index respectively
        """
        # init indices and dicts
        self.index = Index()
        self.index.read_indices()
        with open('pageRank.pkl', 'rb') as f:
            self.page_rank = pickle.load(f)

        with open('pageViews.pkl', 'rb') as f:
            self.page_views = pickle.load(f)

        with open('id2title.pkl', 'rb') as f:
            self.id2title = pickle.load(f)


        print('indices_loaded\nloading all words')
        # init words sets to save time of init them at each search
        self.title_words = set(self.index.titles_index.term_total.keys())
        self.body_words = set(self.index.body_index.term_total.keys())
        self.anchor_words = set(self.index.anchor_index.term_total.keys())
        

    def search(self, query):
        """
        our best search method, make binary search by title and bm25 search on body and then
        merge the results with wieghts, such that 1.5 for title,1 for body, 0.6 for anchor, 0.01 for pagerank.
        and then get the top 100 scores from the merged result
        :param query: str
        :return: list of pairs - [(doc_id, doc_title)...]
        """
        query_tokens = self.index.tokenize(query)
        title_scores = self.title_score(query_tokens)  # convert to format: {query_id: list of (doc_id,score)}
        body_scores = self.search_body_bm25(query_tokens)  # convert to format: {query_id: list of (doc_id,score)}
        anchor_scores = dict(self.anchor_score(query_tokens))
        # return self.search_body_bm25(query_tokens)
        scores = self.merge_results(title_scores, body_scores,anchor_scores, title_weight=1.5, text_weight=1,anchor_weight=0.6, pageRank_weight=0.01, N=100)
        # return scores
        return [(doc_id,self.id2title[doc_id]) for (doc_id, score) in scores]

    def search_by_title(self, query):
        """
        search over the title index for the query according to binary model searching.
        get query string, tokenize it and pass it to self.title_Score to get the title score for each document.
        and then return the ids and titles of the documents
        :param query: str
        :return: list of pairs - [(doc_id, doc_title)...]
        """
        # tokenize query and get score
        query_tokens = self.index.tokenize(query)
        scores = self.title_score(query_tokens)

        return [(doc_id,self.id2title[doc_id]) for (doc_id, score) in scores]

    def search_by_body(self, query):
        """
        search over the body index for the query according to tf-idf + cosine similarity.
        get query string, and pass it to self.new_body_search to get body score for each doc.
        :param query: str
        :return: list of pairs - [(doc_id, doc_title)...]
        """
        res = self.new_body_search(query)
        return [(doc_id,self.id2title[doc_id]) for (doc_id, score) in res]

    def search_by_anchor(self, query):
        """
        search over the anchor index for the query according to binary model searching.
        get query string, tokenize it and pass it to self.anchor_score to get the anchor score for each document.
        and then return the ids and titles of the documents
        :param query: str
        :return: list of pairs - [(doc_id, doc_title)...]
        """
        query_tokens = self.index.tokenize(query)

        res = self.anchor_score(query_tokens)

        titles_res = []
        for (doc_id, score) in res:
            try:
                titles_res.append((doc_id,self.id2title[doc_id]))
            except:
                titles_res.append((0, 'unknown'))
        return titles_res

    def title_score(self, query):
        """
        get the title binary score of each document according to the given query, this func checks only candidate documents
        :param query: list of tokens
        :return: list of tuples: [(doc_id, score) ...] sorted by scores
        """
        score_dict = defaultdict(lambda: 0)
        query = set(query)

        matches = 0

        words = self.title_words

        for term in query:
            # check only terms that in our index
            if term in words:
                # get pls for this term
                list_of_doc = self.index.titles_index.read_posting_list(term)

                # add the binary values to scores dict
                for doc_id, _ in list_of_doc:
                    score_dict[doc_id] += 1

        return sorted(score_dict.items(), key=lambda t: t[1], reverse=True)

    def anchor_score(self, query):
        """
        get the anchor binary score of each document according to the given query, this func checks only candidate documents
        :param query: list of tokens
        :return: list of tuples: [(doc_id, score) ...] sorted by scores
        """
        score_dict = defaultdict(lambda: 0)

        query = set(query)
        matches = 0

        words = self.anchor_words

        for term in query:
            # check only terms that in our index
            if term in words:
                # get pls for this term
                list_of_doc = self.index.anchor_index.read_posting_list(term)

                # add the binary values to scores dictt
                for doc_id, _ in list_of_doc:
                    score_dict[doc_id] += 1

        return sorted(score_dict.items(), key=lambda t: t[1], reverse=True)

    def merge_results(self, title_scores, body_scores, anchor_scores, title_weight=0.5, text_weight=0.5, anchor_weight=0.3,pageRank_weight=0.1, N=3):
        """
        This function merge and sort documents retrieved by its weighte score (e.g., title and body).

        Parameters:
        -----------
        title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                                key: query_id
                                                                                value: list of pairs in the following format:(doc_id,score)

        body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                                key: query_id
                                                                                value: list of pairs in the following format:(doc_id,score)
        title_weight: float, for weigted average utilizing title and body scores
        text_weight: float, for weigted average utilizing title and body scores
        N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

        Returns:
        -----------
        dictionary of querires and topN pairs as follows:
                                                            key: query_id
                                                            value: list of pairs in the following format:(doc_id,score).
        """
        merged_scores_dict = []
        # for q_id, scores in title_scores.items():
        scores_title = dict(title_scores)
        scores_body = dict(body_scores)
        
        docs_id = list(scores_title.keys())
        docs_id.extend(scores_body.keys())
        final_scores = {key: (scores_title.get(key, 0) * title_weight + scores_body.get(key, 0) * text_weight +
                              + self.page_rank[key] * pageRank_weight + anchor_scores.get(key,0)*anchor_weight)
                        for key in docs_id}

        # final_scores = {doc_id: (score + pageRank_weight * self.page_rank[doc_id]) for doc_id, score in
        #                 final_scores.items()}
        merged_scores_dict = sorted(final_scores.items(),
                                    key=lambda x: x[1], reverse=True)[:N]
        return merged_scores_dict

  

    def search_body_bm25(self, query_tokens):
        """
        search in body index using BM25 similarity method, this func is used by the self.search method.

        :param query_tokens: list of str
        :return: list of paits [(doc_id, score)...] sorted by bm25 scores
        """
        # size of the corpus
        size_corpus = len(self.index.body_index.DL)
        idf = {}
        scores = {}
        k1=1.5
        b=0.75
        # aveage doc length
        AVGDL = sum(self.index.body_index.DL.values()) / size_corpus

        def get_candidates_bm25():
            """
            get all candidate docs for query_tokens and calculate the idf and the final bm25 score.
            note that this func was seperated to multiple funcs from hw4 but i merged them for the
            time effeciency.
            :return: list of pars [(doc_id, bm25_score)]
            """

            all_docs_distinct = set()
            term_docid_freq = {}

            # for on all the terms in the query
            for term in query_tokens:
                # if this term in our body index
                if term in self.body_words:
                    # calc query_idf
                    n_ti = self.index.body_index.df[term]
                    idf[term] = np.log(1 + (size_corpus - n_ti + 0.5) / (n_ti + 0.5))
                    
                    list_docid_tf_foreach_term = self.index.body_index.read_posting_list(term)

                    # get pls and calc sore for each doc
                    for doc_id, freq in list_docid_tf_foreach_term:
                        doc_len = self.index.body_index.DL[doc_id]

                        numerator = idf[term] * freq * (k1 + 1)
                        denominator = freq + k1 * (1 - b + b * doc_len / AVGDL)
                        scores[doc_id] = scores.get(doc_id,0) + (numerator / denominator)

            return scores

        doc_id_bm25 = []
        doc_id_bm25 = get_candidates_bm25().items()

        return sorted(doc_id_bm25, key=lambda x: x[1], reverse=True)[:100]

    def get_page_ranks(self, wiki_ids):
        """
        get page rank using self.page_rank dict for each doc id in wiki_ids.
        if id is not in dict return 0
        :param wiki_ids: list of int ids
        :return: list of int ranks
        """
        return [self.page_rank.get(x, 0) for x in wiki_ids]
    
    def get_page_views(self, ids):
        """
        get page views using self.page_views dict for each doc id in wiki_ids.
        if id is not in dict return 0
        :param wiki_ids: list of int ids
        :return: list of int views
        """
        return [self.page_views.get(i,0) for i in ids]

    
    def new_body_search(self, query):
        """
        search in body index using tf-idf and cosine similarity method. this func
        is an upgraded version of the original tf-idf score method so this method is faster and
        dont do extra calculations.
        note that this func makes vector in length of the query vocals and not the entire corpus
        :param query: str
        :return: list of paris [(doc_id, sim_score)...]
        """
        # init some vars
        query_tokens = self.index.tokenize(query)
        query_tf = Counter(query_tokens)
        query_words = np.unique(query_tokens)

        # N = num of docs in corpus
        N = len(self.index.body_index.DL)

        # norma of query(vector length) to calc cosine sim
        query_norma = 0
        tfidf_scores = {}  # doc_id: tfidf score
        for term in query_words:
            # if term is in body index
            if term in self.body_words:
                # calc idf and df
                df = self.index.body_index.df[term]
                idf = np.log10(N / df)

                # get query tfidf value
                query_tfidf = (query_tf[term] / (len(query_tokens) + 0.000001))
                query_norma += query_tfidf ** 2

                pls = self.index.body_index.read_posting_list(term)

                # calc tfidf for each doc in pls and multiply it by query tfidf score to save computing time of another iterations
                for (doc_id, freq) in pls:
                    doc_tfidf = (freq / (self.index.body_index.DL[doc_id] + 0.000001)) * idf
                    summed_tf_idf = doc_tfidf * query_tfidf
                    tfidf_scores[doc_id] = (tfidf_scores.get(doc_id, (0, 0))[0] + summed_tf_idf,
                                            tfidf_scores.get(doc_id, (0, 0))[1] + doc_tfidf ** 2)

        # inner cosine sim func that gets (doc_norma, tfidf score) and return cosine sim with the given query
        def cosine_sim(e):
            tfidf_score, d_norma = e[1]
            return tfidf_score / np.sqrt(d_norma * query_norma)

        # return sorted scores according to cosine sim
        return sorted(tfidf_scores.items(), reverse=True, key=cosine_sim)[:100]

# some tests
if __name__ == "__main__":
    engine = Engine()
    s_time = time()
    # print(engine.index.body_index.posting_locs)
    print(engine.search_by_body('football clubs'))
    # print(engine.search_by_title('football clubs'))
    # print(engine.search('football clubs'))

    # print(engine.title_score(['football', 'clubs', 'hello', 'my']))
    # print(engine.page_rank[3434750])
    # print(engine.id2title)
    # print(engine.get_page_views([17325239]))
    # print(engine.search_by_anchor('america football'))
    # print(engine.index.body_index.read_posting_list('america'))
    # print(engine.index.body_index.DL)
    print(time() - s_time)
