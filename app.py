import streamlit as st
import pandas as pd
import difflib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Page config (MUST be first Streamlit command)

st.set_page_config(
    page_title="Netflix-Style Movie Recommender",
    layout="wide"
)


# Netflix-style CSS

st.markdown("""
<style>
.stApp {
    background-color: #141414;
    color: white;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.movie-card {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.movie-card:hover {
    transform: scale(1.08);
    box-shadow: 0px 15px 40px rgba(0,0,0,0.8);
    z-index: 10;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin: 20px 0 10px 0;
}

button {
    background-color: #e50914 !important;
    color: white !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# Load data

@st.cache_data
def load_data():
    df = pd.read_csv("data/movies_with_cached_posters.csv")
    for col in ["cast", "genres", "director", "keywords", "title"]:
        df[col] = df[col].fillna("").str.lower().str.strip()
    return df

movies = load_data()


# Build similarity matrix

@st.cache_data
def build_similarity(df):
    features = (
        df["genres"] + " " +
        df["keywords"] + " " +
        df["tagline"].fillna("") + " " +
        df["cast"] + " " +
        df["director"]
    )
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(features)
    return cosine_similarity(vectors)

similarity = build_similarity(movies)


# Recommendation logic

def recommend_by_movie(movie_name, top_n=12):
    titles = movies["title"].tolist()
    match = difflib.get_close_matches(movie_name.lower(), titles, n=1)
    if not match:
        return pd.DataFrame()

    idx = movies[movies["title"] == match[0]].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    return movies.iloc[[i[0] for i in scores]]

def filter_movies(actor=None, genre=None, limit=12):
    df = movies.copy()
    if actor:
        df = df[df["cast"].str.contains(actor.lower(), na=False)]
    if genre:
        df = df[df["genres"].str.contains(genre.lower(), na=False)]
    return df.head(limit)


# Netflix-style row renderer

def show_row(title, df, max_items=10):
    if df.empty:
        return

    st.markdown(
        f"<div class='section-title'>{title}</div>",
        unsafe_allow_html=True
    )

    cols = st.columns(max_items)
    for i, (_, row) in enumerate(df.head(max_items).iterrows()):
        with cols[i]:
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)

            poster = row["poster_local"]
            if isinstance(poster, str) and Path(poster).exists():
                st.image(poster, use_column_width=True)

            if st.button(
                row["title"].title(),
                key=f"{title}_{row['id']}"
            ):
                st.session_state.selected_movie = row["title"]
                st.experimental_rerun()

            st.markdown("</div>", unsafe_allow_html=True)


# Session state

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None


# Sidebar

st.sidebar.title("🔍 Search")

movie_input = st.sidebar.text_input("🎬 Movie name")
actor_input = st.sidebar.text_input("🎭 Actor name")
genre_input = st.sidebar.text_input("🎞 Genre")

st.sidebar.markdown("---")


# Netflix-style app logic

if st.session_state.selected_movie:
    base_movie = st.session_state.selected_movie

    st.markdown(
        f"<h2>Because you watched {base_movie.title()}</h2>",
        unsafe_allow_html=True
    )

    similar = recommend_by_movie(base_movie, 10)
    show_row("Similar Movies", similar)

    base_cast = movies.loc[
        movies["title"] == base_movie, "cast"
    ].iloc[0].split(",")[0]

    cast_movies = movies[
        movies["cast"].str.contains(base_cast, na=False)
    ]

    show_row("More From This Cast", cast_movies)

elif movie_input:
    results = recommend_by_movie(movie_input, 12)
    show_row("Search Results", results)

elif actor_input:
    results = filter_movies(actor=actor_input, limit=12)
    show_row(f"Movies Featuring {actor_input.title()}", results)

elif genre_input:
    results = filter_movies(genre=genre_input, limit=12)
    show_row(f"{genre_input.title()} Movies", results)

else:
    show_row(
        "🔥 Trending Now",
        movies.sort_values("vote_average", ascending=False)
    )
    show_row(
        "🎬 Action Movies",
        movies[movies["genres"].str.contains("action")]
    )
    show_row(
        "😂 Comedy Picks",
        movies[movies["genres"].str.contains("comedy")]
    )
    show_row(
        "🧠 Drama Spotlight",
        movies[movies["genres"].str.contains("drama")]
    )
