#Import modules
import numpy as np
import pandas as pd 
import nltk
import re
import datamanipulation as dm

#set rand seed
np.random.seed(6)

#read in test movie data
movies_df = pd.read_csv("movies.csv")

print("Num of movies loaded: %s " % (len(movies_df)))

#combine plot summaries into single column
movies_df["plot"] = movies_df["wiki_plot"].astype(str) + "\n" \
    + movies_df["imdb_plot"].astype(str)

#create tfidf matrix with vectorizer function, which also uses tokenize_and_stem
tfidf_matrix = dm.vectorizer(movies_df)

#create clusters with KMeans function
movies_df["cluster"] = dm.create_clusters(tfidf_matrix)

#call function to calculate similarity distance
sim_dist = dm.similarity_distance(tfidf_matrix)

#call function to create dendrogram visualization
dm.create_dendrogram(sim_dist, movies_df)