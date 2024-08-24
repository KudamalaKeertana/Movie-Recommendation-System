# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__, static_url_path='', static_folder='')

# Load the dataset from CSV file
df = pd.read_csv('E:\ML\movies.csv')

# Preprocessing: Combine title and genres into one string
df['title_genres'] = df['title'] + ' ' + df['genres']

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title_genres'])

# Initialize the Nearest Neighbors model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(tfidf_matrix)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommend_movies():
    data = request.get_json()
    genres_input = data.get('genres', '')  # Get 'genres' from JSON data, default to empty string
    
    # Check if genres_input is empty or contains invalid characters
    if not genres_input:
        return jsonify({"error": "Please enter correct genre values."}), 400
    
    def recommend_movies_knn(genres, df, tfidf_matrix, model, n=5):
        # Split the genres string into individual genres
        genre_list = genres.split(',')

        # Combine the new genres with the title for the new movie
        new_movie = "New Movie " + ','.join(genre_list)

        # Transform the new movie's genres using the trained TF-IDF vectorizer
        new_movie_tfidf = tfidf_vectorizer.transform([new_movie])

        # Find the k-nearest neighbors
        _, indices = model.kneighbors(new_movie_tfidf, n_neighbors=n+1)

        # Get top n recommended movies (excluding the new movie itself)
        top_movies_indices = indices.flatten()[1:]
        recommended_movies = df.iloc[top_movies_indices]['title'].tolist()

        return recommended_movies
    
    recommended_movies = recommend_movies_knn(genres_input, df, tfidf_matrix, knn_model)
    return jsonify(recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
