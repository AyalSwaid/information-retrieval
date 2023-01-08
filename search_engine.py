from build_index import Index
from collections import Counter, OrderedDict, defaultdict
from time import time

# TODO: try to search using pyspark, this TODO should be the last one to be done

class Engine:
    def __init__(self):
        self.index = Index()
        self.index.read_indices()

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



        # for w, pls in self.index.titles_index.posting_lists_iter():
        #     # print(w)
        #     if w in query:
        #         matches += 1
        #         for doc_id, _ in pls:
        #             score_dict[doc_id] += 1
        #     if matches == len(query):
        #         break



        return sorted(score_dict.items(), key=lambda t: t[1], reverse=True)

    def body_score(self, query):
        words = set(self.index.body_index.term_total)
        scores = defaultdict(lambda : 0)

        for term in query:
            if term in words:
                list_of_doc = self.index.titles_index.read_posting_list(term)
                for doc_id, freq in list_of_doc:
                    scores[doc_id] += freq

        return sorted(scores.items(), key=lambda t: t[1], reverse=True)

if __name__ == "__main__":
    engine = Engine()
    s_time = time()
    # print(engine.index.body_index.posting_locs)
    print(engine.search_by_body('football clubs'))
    print(engine.search_by_title('football clubs'))
    print(time() - s_time)
