# information-retrieval
Information retrieval project that process the wikipedia enlish corpus and uses methods like inverted indices, tf-idf, cosine similarity, BM25 on the title and body text of each wiki page.
# files included 
we included files to build, read and use the indices. also we uploaded a small corpus example of our indices in the files anchor_index, body_index, title_index.

# build the index
* build_index.ipynb - notebook that we built the index step by step and uploaded the complete index on the GCP storage, not that this notebook is used only on a GCP jupyter notebook because it processes a large corpus of data.
* inverted_index_gcp.py/inverted_index_colab.py - files that contain the InvertedIndex class, the inverted_index_gcp is adjusted to work with the GCP storage.

# Read And Use The Index To Search
* inverted_index_gcp.py/inverted_index_colab.py - files that contain the InvertedIndex class, the inverted_index_gcp is adjusted to work with the GCP storage.
* search_frontend.py - the flask app file that gets queries from the internet, and then uses search_engine.Engine to perform all the search methods.
* search_engine.py - the main engine functions is in this file, contains the Engine class which has the search methods and has access to the Index object from build_index.py.
* build_index.py - this file contains the Index class which is the only class that has direct access to the built inverted indices(using class inverted_index_gcp.InvertedIndex). Index class contains the three inverted indices: body_index, title_index, anchor_index.
* run_fronted_colab.ipynb - runs(or connect to) the lask app and do some query tests and measure the MAP score.
* run_engine_updated.ipynb - also a note book to test the flask app but it is used in GCP notebook because it loads the indices files(10GB) into local disk and run the engine or flask app there.

# Notes:
* you probably cant run the search engine because we did not upload all the files like pageViews.pkl and othe global statistics files because of their size.
* do not confuse with build_index.py, it does not build the inverted index but it does read the built inverted index.

# A Diagram that clearifies how our retrieval system works:
![IRprojUML drawio (1)](https://user-images.githubusercontent.com/57876635/212572470-93ce2915-f3fe-4ccb-a535-fc9b397f07db.png)

