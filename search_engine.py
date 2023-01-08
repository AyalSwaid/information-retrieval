from build_index import Index
from collections import Counter, OrderedDict, defaultdict
from time import time

# TODO: try to search using pyspark, this TODO should be the last one to be done

class Engine:
    def __init__(self):
        self.index = Index()
        self.index.read_indices()

    def search(self, query):
        query_tokens = self.index.tokenize(query)
        title_scores = {0: self.title_score(query_tokens)} # convert to format: {query_id: list of (doc_id,score)}
        body_scores = {0: self.body_score(query_tokens)} # convert to format: {query_id: list of (doc_id,score)}

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


if __name__ == "__main__":
    engine = Engine()
    s_time = time()
    # print(engine.index.body_index.posting_locs)
    # print(engine.search_by_body('football clubs'))
    # print(engine.body_score(['america']))
    print(engine.search_by_anchor('america football'))
    # print(engine.index.body_index.read_posting_list('america'))
    print(time() - s_time)
