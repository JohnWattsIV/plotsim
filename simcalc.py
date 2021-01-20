#Import modules
import numpy as np
import pandas as pd 
import nltk

#set rand seed
np.random.seed(6)

#read in test movie data
movies_df = pd.read_csv("movies.csv")

print("Num of movies loaded: %s " % (len(movies_df)))

#display loaded data
print(movies_df)