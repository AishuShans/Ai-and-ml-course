import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
books = pd.read_csv("books.csv")
ratings = pd.read_csv("ratings.csv")

# ----------- Content-Based Filtering Setup -----------
books['features'] = books['Title'] + ' ' + books['Author'] + ' ' + books['Genre']
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['features'])
content_similarity = cosine_similarity(tfidf_matrix)

# ----------- Collaborative Filtering Setup -----------
user_item_matrix = ratings.pivot_table(index='User_ID', columns='Book_ID', values='Rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# ----------- Streamlit UI -----------
st.title("📚 Book Recommendation System")
st.write("Choose your recommendation method:")

option = st.selectbox("Select Recommendation Type", ["Content-Based", "Collaborative", "Hybrid"])

if option == "Content-Based":
    selected_title = st.selectbox("Choose a Book Title", books['Title'])
    book_index = books[books['Title'] == selected_title].index[0]
    similar_books = content_similarity[book_index].argsort()[::-1][1:4]
    st.subheader("📖 Recommended Books (Content-Based):")
    for i in similar_books:
        st.write(books.iloc[i]['Title'])

elif option == "Collaborative":
    user_id = st.number_input("Enter User ID", min_value=int(ratings['User_ID'].min()),
                              max_value=int(ratings['User_ID'].max()), step=1)
    if user_id in user_sim_df.index:
        similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:4]
        st.subheader("👥 Most Similar Users:")
        for uid in similar_users.index:
            st.write(f"User ID: {uid} (Similarity Score: {similar_users[uid]:.2f})")
    else:
        st.warning("User ID not found in dataset.")

elif option == "Hybrid":
    selected_title = st.selectbox("Choose a Book Title", books['Title'], key="hybrid")
    user_id = st.number_input("Enter User ID for Hybrid", min_value=int(ratings['User_ID'].min()),
                              max_value=int(ratings['User_ID'].max()), step=1, key="hybrid_user")

    if user_id in user_item_matrix.index:
        book_index = books[books['Title'] == selected_title].index[0]
        content_scores = content_similarity[book_index]
        user_ratings = user_item_matrix.loc[user_id]
        aligned_ratings = user_ratings.reindex(books['Book_ID']).fillna(0).values
        hybrid_score = 0.6 * content_scores + 0.4 * aligned_ratings
        top_indices = np.argsort(hybrid_score)[::-1]
        recommended_indices = [i for i in top_indices if i != book_index][:3]

        st.subheader("🔀 Hybrid Recommendations:")
        for i in recommended_indices:
            st.write(books.iloc[i]['Title'])
    else:
        st.warning("User ID not found in dataset.")
