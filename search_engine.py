from build_index import Index
from collections import Counter, OrderedDict, defaultdict
from time import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# TODO: try to search using pyspark, this TODO should be the last one to be done

class Engine:
    def __init__(self):
        self.index = Index()
        self.index.read_indices()
        self.page_rank = pd.read_csv('pageRank.csv', index_col=0, names=['rank']).to_dict()['rank']# {doc_id:rank}
        # self.id2title = pd.read_csv('id2title.csv', index_col=0, names=['title']).to_dict()['title']# {doc_id:title}

    def search(self, query):
        query_tokens = self.index.tokenize(query)
        title_scores = {0: self.title_score(query_tokens)} # convert to format: {query_id: list of (doc_id,score)}
        body_scores = {0: self.search_body_bm25(query_tokens)} # convert to format: {query_id: list of (doc_id,score)}
        # return self.search_body_bm25(query_tokens)
        return self.merge_results(title_scores, body_scores, title_weight=0.4, text_weight=0.2, N=100)

    def search_by_title(self, query):
        # tokenize query
        query_tokens = self.index.tokenize(query)
        scores = self.title_score(query_tokens)

        # TODO: get the title for each doc id from scores

        return scores

    def search_by_body(self, query):
        query_tokens = self.index.tokenize(query)

        res = self.body_score(query_tokens)

        return res

    def search_by_anchor(self, query):
        query_tokens = self.index.tokenize(query)

        res = self.anchor_score(query_tokens)

        return res

    def posting_iter(self, index):
        """
        This function returning the iterator working with posting list.

        Parameters:
        ----------
        index: inverted index
        """
        # TODO: check if delete this method
        words, pls = zip(*index.posting_lists_iter())
        return words, pls

    def title_score(self, query):
        """
        TODO: docs
        :param query: str
        :param doc_word_count:
        :return:
        """
        b_start = time()
        score_dict = defaultdict(lambda : 0)

        query = set(query)
        matches = 0
        # words, ppp = self.posting_iter(self.index.titles_index)

        # TODO: may init words in self.__init__ to save time
        words = set(self.index.titles_index.term_total.keys())

        for term in query:
            if term in words:
                list_of_doc = self.index.titles_index.read_posting_list(term)
                for doc_id, _ in list_of_doc:
                    score_dict[doc_id] += 1
        print('title score time: ', time() - b_start)


        return sorted(score_dict.items(), key=lambda t: t[1], reverse=True)

    def body_score(self, query):
        """

        :param query: tokenized
        :return:
        """

        docs_tfidf = self.generate_document_tfidf_matrix(query)
        query_tfidf = self.generate_query_tfidf_vector(query)
        cosine_sim = self.cosine_sim_using_sklearn(query_tfidf,docs_tfidf)

        return sorted([(doc_id,np.round(score,5)) for doc_id, score in cosine_sim.items()], key = lambda x: x[1],reverse=True)[:100]

    def body_score_tf(self, query):
        words = set(self.index.body_index.term_total)
        scores = {}

        for term in query:
            if term in words:
                list_of_doc = self.index.body_index.read_posting_list(term)
                for doc_id, freq in list_of_doc:
                    scores[doc_id] = scores.get(doc_id, 0) + freq
        return sorted(scores.items(), key=lambda t: t[1], reverse=True)[:100]

    def anchor_score(self, query):
        """
                TODO: docs
                :param query: tokens
                :param doc_word_count:
                :return:
                """
        b_start = time()
        score_dict = defaultdict(lambda: 0)

        query = set(query)
        matches = 0
        # words, ppp = self.posting_iter(self.index.titles_index)

        # TODO: may init words in self.__init__ to save time
        words = set(self.index.anchor_index.term_total.keys())

        for term in query:
            if term in words:
                list_of_doc = self.index.anchor_index.read_posting_list(term)
                for doc_id, _ in list_of_doc:
                    score_dict[doc_id] += 1
        print('title score time: ', time() - b_start)

        return sorted(score_dict.items(), key=lambda t: t[1], reverse=True)

    def generate_query_tfidf_vector(self, query_to_search):
        """
        Generate a vector representing the query. Each entry within this vector represents a tfidf score.
        The terms representing the query will be the unique terms in the index.

        We will use tfidf on the query as well.
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the query.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        Returns:
        -----------
        vectorized query with tfidf scores
        """

        epsilon = .0000001
        total_vocab_size = len(self.index.body_index.term_total)
        Q = np.zeros((total_vocab_size))
        term_vector = list(self.index.body_index.term_total.keys())
        counter = Counter(query_to_search)
        for token in np.unique(query_to_search):
            if token in term_vector:  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
                df = self.index.body_index.df[token]
                idf = np.log10((len(self.index.body_index.DL)) / (df + epsilon))  # smoothing

                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q

    def get_candidate_documents_and_scores(self, query_tokens):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']
        index:           inverted index loaded from the corresponding files.
        words,pls: generator for working with posting.
        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        words = list(self.index.body_index.term_total.keys())

        candidates = {}
        for term in np.unique(query_tokens):
            if term in words:
                list_of_doc = self.index.body_index.read_posting_list(term)
                normlized_tfidf = [(doc_id, (freq / self.index.body_index.DL[doc_id]) * np.log(len(self.index.body_index.DL) / self.index.body_index.df[term])) for
                                   doc_id, freq in list_of_doc]

                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates

        # DL = self.index.body_index.DL
        # # query_tokens = self.index.tokenize(query_to_search)
        # query_counter = Counter(query_tokens)
        # query_counterd = dict(query_counter)
        # norma_q = 0
        # norma_d = 0
        # candidates = {}
        # N = len(DL)
        # for term in np.unique(query_tokens):
        #     if term in words:
        #         norma_q += query_counterd[term]**2
        #         list_of_doc = self.index.body_index.read_posting_list(term)
        #         normlized_tfidf = []
        #         for doc_id, freq in list_of_doc:
        #             if (doc_id, freq) == (0, 0):
        #                 continue
        #
        #             formula = (freq / DL[doc_id]) * np.log10(N / self.index.body_index.df[term])*query_counterd[term]
        #             id_tfidf = (doc_id, formula)
        #             normlized_tfidf.append(id_tfidf)
        #
        #         for doc_id, tfidf in normlized_tfidf:
        #             tfidf_val, norma = candidates.get(doc_id,(0, 0))
        #             candidates[doc_id] = (tfidf_val + tfidf, norma + tfidf**2)
        #
        # return candidates, norma_q

    def generate_document_tfidf_matrix(self, query_to_search):
        """
        Generate a DataFrame `D` of tfidf scores for a given query.
        Rows will be the documents candidates for a given query
        Columns will be the unique terms in the index.
        The value for a given document and term will be its tfidf score.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.


        words,pls: iterator for working with posting.

        Returns:
        -----------
        DataFrame of tfidf scores.
        """

        total_vocab_size = len(self.index.body_index.term_total)
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), total_vocab_size))
        D = pd.DataFrame(D)

        D.index = unique_candidates
        D.columns = self.index.body_index.term_total.keys()

        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key

            D.loc[doc_id][term] = tfidf

        return D

    def cosine_sim_using_sklearn(self, query, tfidf):
        """
        In this function you need to utilize the cosine_similarity function from sklearn.
        You need to compute the similarity between the queries and the given documents.
        This function will return a DataFrame in the following shape: (# of queries, # of documents).
        Each value in the DataFrame will represent the cosine_similarity between given query and document.

        Parameters:
        -----------
          queries: sparse matrix represent the queries after transformation of tfidfvectorizer.
          documents: sparse matrix represent the documents.

        Returns:
        --------
          DataFrame: This function will return a DataFrame in the following shape: (# of queries, # of documents).
          Each value in the DataFrame will represent the cosine_similarity between given query and document.
        """
        # queries = queries.reshape(1, -1)
        # query_geo_size = sum([i ** 2 for i in query])
        query_geo_size = np.dot(query, query)
        def sim(d):
            return (np.dot(np.array(d), query)) / np.sqrt((query_geo_size * np.dot(d,d)))

        sims = tfidf.apply(sim, axis=1)
        return sims.to_dict()

    def merge_results(self, title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
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
        merged_scores_dict = {}
        for q_id, scores in title_scores.items():
            scores_title = dict(scores)
            scores_body = dict(body_scores[q_id])
            docs_id = list(scores_title.keys())
            docs_id.extend(scores_body.keys())
            final_scores = {key: scores_title.get(key, 0) * title_weight + scores_body.get(key, 0) * text_weight for key
                            in docs_id}
            merged_scores_dict[q_id] = sorted([(doc_id, score) for doc_id, score in final_scores.items()],
                                              key=lambda x: x[1], reverse=True)[:N]
        return merged_scores_dict

    def get_page_ranks(self, wiki_ids):
        return [self.page_rank.get(x,0) for x in wiki_ids]

    def search_body_bm25(self, query_tokens):
        # TODO: change salman comments
        # size of the corpus
        size_corpus = len(self.index.body_index.DL)

        # aveage doc length
        AVGDL = sum(self.index.body_index.DL.values()) / size_corpus

        def get_candidates_bm25():
            # tokens = tokenize(queries)
            all_docs_distinct = set()
            term_docid_freq = {}

            # for on all the terms in the query
            for term in query_tokens:
                if term in self.index.body_index.term_total:
                    list_docid_tf_foreach_term = self.index.body_index.read_posting_list(term)
                    # lst_docid = []

                    # getting a list of doc id to each term
                    # getting a dictionary that to each (doc_id,term) his tf-term frequency
                    for doc_id, freq in list_docid_tf_foreach_term:
                        term_docid_freq[(term, doc_id)] = freq
                        # lst_docid.append(doc_id)

                        all_docs_distinct = all_docs_distinct | {doc_id}

                # getting only distinct docs
            # all_docs_distinct = set(all_docs_distinct)
            return term_docid_freq, all_docs_distinct

        def bm25_score(doc_id, freqs, k1=1.5, b=0.75):
            """
                    This function calculate the bm25 score for given query and document.

                    Parameters:
                    -----------
                    query: list of token representing the query. For example: ['look', 'blue', 'sky']
                    doc_id: integer, document id.

                    Returns:
                    -----------
                    score: float, bm25 score.
                    """
            score = 0.0
            doc_len = self.index.body_index.DL[doc_id]
            idf = calc_idf(query_tokens)

            for term in query_tokens:
                if (term,doc_id) in freqs:
                    freq = freqs[(term, doc_id)]

                    # freq = term_frequencies[doc_id]
                    numerator = idf[term] * freq * (k1 + 1)
                    denominator = freq + k1 * (1 - b + b * doc_len / AVGDL)
                    score += (numerator / denominator)
            return score

        def calc_idf(list_of_tokens):
            """
            This function calculate the idf values according to the BM25 idf formula for each term in the query.

            Parameters:
            -----------
            query: list of token representing the query. For example: ['look', 'blue', 'sky']

            Returns:
            -----------
            idf: dictionary of idf scores. As follows:
                                                        key: term
                                                        value: bm25 idf score
            """
            idf = {}
            for term in list_of_tokens:
                if term in self.index.body_index.df.keys():
                    n_ti = self.index.body_index.df[term]
                    idf[term] = np.log(1 + (size_corpus - n_ti + 0.5) / (n_ti + 0.5))
                else:
                    pass
            return idf

        doc_id_bm25 = []
        freqs,docs_ids = get_candidates_bm25()
        # return freqs
        for doc_id in docs_ids:
            doc_id_bm25.append((doc_id, bm25_score(doc_id, freqs, 1.5, 0.75)))

        return sorted(doc_id_bm25, key=lambda x: x[1], reverse=True)[:100]


if __name__ == "__main__":
    engine = Engine()
    s_time = time()
    # print(engine.index.body_index.posting_locs)
    print(engine.search('football clubs'))
    # print(engine.body_score(['america']))
    # print(engine.search_by_anchor('america football'))
    # print(engine.index.body_index.read_posting_list('america'))
    # print(engine.index.body_index.DL)
    print(time() - s_time)
