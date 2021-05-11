from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
import numpy as np
import os
import pickle
import sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

app = FastAPI()
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.fillna(0,inplace=True)
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

pkl_filename = "knn_model.pkl"
with open(pkl_filename, 'rb') as file:
    knn_ = pickle.load(file)

def get_movie_recommendation(movie_name,movies,knn):
    n_movies_to_reccomend = 100
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"

def multiple_movie(list_movie,movies,knn,top_x):
    # print(list_movie)
    try:
        list_df_temp = []
        if len(list_movie) == 1:
            return {
                'status':'success',
                'recomendation_movies':get_movie_recommendation(list_movie[0],movies,knn).head(top_x).reset_index(drop=True)}
        else:
          for i in range(len(list_movie)):
              if i == 0:
                  list_df_temp.append(get_movie_recommendation(list_movie[i],movies,knn))
              else:
                  list_df_temp.append(get_movie_recommendation(list_movie[i],movies,knn))
                  df_temp = pd.concat(list_df_temp)
                  df_temp = df_temp.groupby(df_temp.Title).mean().reset_index()
                  list_df_temp = [df_temp]
          return {
              'status':'success',
              'recomendation_movies': df_temp.sort_values(by=['Distance'],ascending=False).head(top_x).reset_index(drop=True)}
    except:
          return {'status':'error','msg':'invalid input'}

@app.get("/")
def read_item():
    """return active endpoint"""
    return {'1':'/predict_api/','2':'/user_predict/'}

@app.get("/predict_api/")
async def predict_api(list_movie: str = 'Iron Man',top_x: str = '5'):
    """input movies, separate by ;, ex: Iron Man;Memento"""
    result = multiple_movie(list_movie.split(';'),movies,knn_,int(top_x))
    # print(result)
    if result['status'] == 'success':
        result['recomendation_movies'] = result['recomendation_movies'].to_dict('index')
        result['list_movies'] = list_movie.split(';')
        result2 = {}
        result2['status'] = result['status']
        result2['your_list_movies'] = result['list_movies']
        result2['recomendation_movies'] = result['recomendation_movies']
        # print(tes)
        # print(list_movie.split(';'))
        # result['list']
        return result2
    else:
        return result

@app.get("/user_predict/")
async def user_predict(user: str = '0',top_x: str = '5'):
    """input 0 to 999"""
    df = pd.read_csv('user_dummy.csv').groupby(['user'])['mov_fav'].apply(lambda x: ';'.join(x)).reset_index()
    df = df[df['user']==int(user)]
    list_movie = df['mov_fav'][df.index[0]]
    # print(list_movie,list_movie.split(';'))
    result = multiple_movie(list_movie.split(';'),movies,knn_,int(top_x))
    # print(result)
    if result['status'] == 'success':
        result['recomendation_movies'] = result['recomendation_movies'].to_dict('index')
        result['list_movies'] = list_movie.split(';')
        result2 = {}
        result2['status'] = result['status']
        result2['user_id'] = user
        result2['user_list_movies'] = result['list_movies']
        result2['recomendation_movies'] = result['recomendation_movies']
        # print(list_movie.split(';'))
        # result['list']
        return result2
    else:
        return result

