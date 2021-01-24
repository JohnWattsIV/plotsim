#import modules
import numpy as np
import pandas as pd 
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

#function to tokenize and stem text
def tokenize_and_stem(text):
    
    #tokenize the summary into sentences and then words using list comprehension
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    #Filter out raw tokens and other non words
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]

    #stem the filtered tokens
    stemmer = SnowballStemmer("english")

    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems

#function to create tfidf vectorizer and fit transform
def vectorizer(text):
    #create tfidfvectorizer object with stopwords and parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, stop_words='english', use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    
    #create vector representation of the plot summaries using fit transform
    tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in text["plot"]])

    return tfidf_matrix

#function to create clusters based on similarity
def create_clusters(matrix):

    #create KMeans object and set desired number of clusters
    km = KMeans(n_clusters = 5)

    #fit object with tfidf_matrix
    km.fit(matrix)

    #set clusters as list
    clusters = km.labels_.tolist()
    
    return clusters
    
#calculate the similarity distance
def similarity_distance(matrix):

    similarity_dist = 1 - cosine_similarity(matrix)

    return similarity_dist

#create a dendrogram of plot similarity
def create_dendrogram(simdist, moviesdf):

    # Create mergings matrix 
    merged = linkage(simdist, method='complete')

    # Plot the dendrogram, using title as label column
    dendrogram_ = dendrogram(merged, labels=[x for x in moviesdf["title"]], leaf_rotation=90, leaf_font_size=16)

    # Adjust the plot
    fig = plt.gcf()
    _ = [lbl.set_color('#000000') for lbl in plt.gca().get_xmajorticklabels()]
    fig.set_size_inches(108,21)

    # Show the plotted dendrogram
    plt.savefig('dendro.png', format = 'png', bbox_inches = 'tight')
    plt.show()

#function to create CSV files in a format that GEPHI can read
def gephi_csv(sim_array, movie_frame):

    #create node list csv with id, lavel, node size, cluster
    node_list = pd.DataFrame(columns = ['id', 'title', 'count', 'modularity'])
    node_list['id'] = movie_frame['rank']
    node_list['title'] = movie_frame['title']
    node_list['count'] = 100/(movie_frame['rank'] + 1)
    node_list['modularity'] = movie_frame['cluster']

    #send node list to csv
    node_list.to_csv('nodelist.csv', index = False)


    #create edge list csv with source node, target node, weight(10/sim score)
    edge_list = pd.DataFrame(columns = ['source', 'target', 'weight'])
    
    #set index variable
    i = 0

    #for every source node
    for index, arr in enumerate(sim_array):

        #for every target node of the source
        for jndex, value in enumerate(arr):

            #source and target cannot be the same node or an edge already added
            if index < jndex:
                #add the edge to the edge list
                weight = 10 / value
                edge_list.loc[i] = [index, jndex, weight]
                i += 1

    #send edge list to csv
    edge_list.to_csv('edgelist.csv', index = False)                