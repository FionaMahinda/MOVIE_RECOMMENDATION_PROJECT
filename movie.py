import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the datasets
df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')

# Merging the datasets on 'id' column
df1.columns = ['id', 'title', 'cast', 'crew']
df2 = df2.merge(df1, on='id')

# Preprocessing steps (if any)
# ...

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df2['overview'].fillna(''))

# Cosine Similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df2[df2['title'] == title].index[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]

# Streamlit App
def main():
    st.title('Movie Recommendation App')
    
    # Sidebar for user input
    user_input = st.text_input('Enter a movie title:', 'The Dark Knight Rises')
    
    # Get recommendations based on user input
    if st.button('Get Recommendations'):
        recommendations = get_recommendations(user_input)
        st.write("Recommended Movies:")
        st.write(recommendations)

if __name__ == "__main__":
    main()
