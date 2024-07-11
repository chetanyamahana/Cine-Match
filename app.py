from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = Flask(__name__)

# Load and preprocess datasets
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')
rs = pd.merge(movies, credits, on='title')

# Functions for preprocessing
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

# Apply preprocessing
rs['keywords'] = rs['keywords'].apply(convert)
rs['genres'] = rs['genres'].apply(convert)
rs['crew'] = rs['crew'].apply(fetch_director)
rs['cast'] = rs['cast'].apply(convert3)
rs['overview'] = rs['overview'].str.lower()
rs = rs[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
rs.dropna(inplace=True)

# Combine features into a single string
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['genres']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['crew']) + ' ' + x['overview']

rs['soup'] = rs.apply(create_soup, axis=1)

# Create count matrix and cosine similarity matrix
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(rs['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset index of DataFrame and construct reverse mapping
rs = rs.reset_index()
indices = pd.Series(rs.index, index=rs['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return rs['title'].iloc[movie_indices]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie']
    recommendations = get_recommendations(movie_title)
    return render_template('recommendations.html', title=movie_title, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
