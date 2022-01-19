#Importing Libraries
import numpy as np
import pandas as pd
import streamlit as st
import os

st.title('What do you want to watch tonight ? 🎬')

def load_data():
    data = pd.io.parsers.read_csv('ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    encoding='unicode_escape',                          
    engine='python', delimiter='::')
    return data

def load_movie():
    mdata = pd.io.parsers.read_csv('movies.dat',
    names=['movie_id', 'title', 'genre'],
    encoding='unicode_escape',                                
    engine='python', delimiter='::')
    return mdata

data = load_data()
movie_data = load_movie()

if os.path.exists("svd_matrix.npy"):
    svd_matrix = np.load("svd_matrix.npy")
else:
    #Creating the rating matrix (rows as movies, columns as users)
    ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
    ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values
    #Normalizing the matrix(subtract mean off)
    normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T
    #Computing the Singular Value Decomposition (SVD)
    A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
    U, S, V = np.linalg.svd(A)
    svd_matrix = V
    np.save("svd_matrix", svd_matrix)
            

#Function to calculate the cosine similarity (sorting by most similar and returning the top N)
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Function to find movie
def findmovie( matrix, movie_id, nr ) :
    #k-principal components to represent movies, movie_id to find recommendations, top_n print n results        
    k = 50
    sliced = matrix.T[:, :k] # representative data
    indexes = top_cosine_similarity(sliced, movie_id, nr)
    return indexes
    
# Function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    st.header("**Movie Recommendation for {}**".format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    st.subheader("***Genre : {}***".format(movie_data[movie_data.movie_id == movie_id].genre.values[0]))
    for id in top_indexes + 2:
        st.write(movie_data[movie_data.movie_id == id].title.values[0], movie_data[movie_data.movie_id == id].genre.values[0] )

#-- Select movie
select_movie = st.sidebar.selectbox('Select/Search your movie',
                                    movie_data["title"])

#-- Select nr of result
nr = st.sidebar.slider('Nr of Result', 10, 20)

rslt_df = movie_data[movie_data['title'] == select_movie]
movie_id =  rslt_df["movie_id"].values[0]

result_i = findmovie(svd_matrix, movie_id, nr )

#Printing the top N similar movies
print_similar_movies(movie_data, movie_id, result_i)
