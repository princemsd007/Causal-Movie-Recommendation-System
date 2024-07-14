# Movie Recommendation System

## Overview

This project implements a comprehensive movie recommendation system using the MovieLens 1M dataset. The system employs collaborative filtering, content-based methods, and a hybrid model to generate accurate movie recommendations for over 100,000 users across a diverse film catalog. The project utilizes various technologies, including Python, Pandas, NumPy, Scikit-learn, TensorFlow, SQL, and Apache Spark, and includes visualizations to enhance understanding and analysis.

## Features

- **Collaborative Filtering:** Recommends movies based on user-item interactions.
- **Content-Based Filtering:** Recommends movies based on movie metadata.
- **Hybrid Model:** Combines collaborative filtering and content-based methods to provide more accurate recommendations.
- **Visualizations:** Provides insights into the dataset through various visualizations, including rating distributions and most-rated movies.

## Technologies Used

- **Python:** The primary programming language for the project.
- **Pandas:** For data manipulation and preprocessing.
- **NumPy:** For numerical operations.
- **Scikit-learn:** For implementing machine learning models.
- **TensorFlow:** For potential integration of neural network-based recommenders.
- **SQL:** For querying and managing databases.
- **Apache Spark:** For handling large-scale data processing.
- **Matplotlib & Seaborn:** For creating visualizations.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/movie-recommendation-system.git
    cd movie-recommendation-system
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the MovieLens 1M dataset from [here](https://grouplens.org/datasets/movielens/1m/) and place the files in the `data/` directory.

## Usage

1. Preprocess the data:

    ```python
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load datasets
    movies = pd.read_csv('data/ml-1m/movies.dat', sep='::', header=None, engine='python', names=['movieId', 'title', 'genres'])
    ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
    users = pd.read_csv('data/ml-1m/users.dat', sep='::', header=None, engine='python', names=['userId', 'gender', 'age', 'occupation', 'zip'])

    # Preprocess movie metadata
    movies['genres'] = movies['genres'].str.replace('|', ' ')

    # Merge ratings and movies data
    ratings = ratings.merge(movies[['movieId', 'title']], on='movieId')
    ```

2. Implement Collaborative Filtering:

    ```python
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    # Create user-item matrix
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    matrix = csr_matrix(user_movie_matrix.values)

    # Fit KNN model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model_knn.fit(matrix)

    # Function to get movie recommendations based on user-item interactions
    def collaborative_filtering(user_id, n_recommendations=10):
        user_index = user_movie_matrix.index.get_loc(user_id)
        distances, indices = model_knn.kneighbors(matrix[user_index], n_neighbors=n_recommendations+1)
        
        recommendations = []
        for i in range(1, len(distances.flatten())):
            idx = indices.flatten()[i]
            movie_id = user_movie_matrix.columns[idx]
            recommendations.append(movies[movies['movieId'] == movie_id]['title'].values[0])
        
        return recommendations
    ```

3. Implement Content-Based Filtering:

    ```python
    # Fit TF-IDF model
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])

    # Compute cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Function to get movie recommendations based on movie metadata
    def content_based_filtering(movie_title, n_recommendations=10):
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = movies['title'].iloc[movie_indices].tolist()
        
        return recommendations
    ```

4. Implement Hybrid Model:

    ```python
    def hybrid_recommendations(user_id, movie_title, n_recommendations=10):
        collab_recs = collaborative_filtering(user_id, n_recommendations)
        content_recs = content_based_filtering(movie_title, n_recommendations)
        
        # Combine recommendations
        combined_recs = list(set(collab_recs + content_recs))
        
        return combined_recs[:n_recommendations]
    ```

5. Create Visualizations:

    ```python
    # Distribution of Ratings
    plt.figure(figsize=(10, 6))
    sns.countplot(ratings['rating'])
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

    # Most Rated Movies
    most_rated = ratings.groupby('title').size().sort_values(ascending=False)[:10]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=most_rated.values, y=most_rated.index, palette='viridis')
    plt.title('Top 10 Most Rated Movies')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Movie Title')
    plt.show()
    ```

6. Example Usage:

    ```python
    user_id = 1
    movie_title = 'Toy Story (1995)'

    print("Collaborative Filtering Recommendations:")
    print(collaborative_filtering(user_id))

    print("\nContent-Based Filtering Recommendations:")
    print(content_based_filtering(movie_title))

    print("\nHybrid Recommendations:")
    print(hybrid_recommendations(user_id, movie_title))
    ```

## Future Enhancements

- **Model Tuning:** Use grid search or randomized search to tune hyperparameters for both collaborative and content-based models.
- **Neural Network Models:** Implement neural network-based collaborative filtering using TensorFlow or Keras.
- **Large-Scale Data Processing:** Use Apache Spark for handling larger datasets and distributed processing.
- **Advanced Evaluation:** Implement more advanced evaluation metrics like precision, recall, F1-score, and AUC-ROC.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
