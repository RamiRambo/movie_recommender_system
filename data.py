"""This module features functions and classes to manipulate data for the
collaborative filtering algorithm.
"""

from pathlib import Path

import scipy
import pandas as pd


def load_reviewer_movie(reviewer_movie_file: Path) -> scipy.sparse.csr_matrix:
    """Load the reviewer movie file and return a reviewer-movie matrix in csr
    fromat.
    """
    reviewer_movie = pd.read_csv(reviewer_movie_file, sep=",")
    reviewer_movie.set_index(["userID", "movieID"], inplace=True)
    coo = scipy.sparse.coo_matrix(
        (
            reviewer_movie.rating.astype(float),
            (
                reviewer_movie.index.get_level_values(0),
                reviewer_movie.index.get_level_values(1),
            ),
        )
    )
    return coo.tocsr()


class MovieRetriever:
    """The MovieRetriever class gets the movie name from the movie ID."""

    def __init__(self):
        self._movie_df = None

    def get_movie_name_from_id(self, movie_id: int) -> str:
        """Return the movie name from the movie ID."""
        return self._movie_df.loc[movie_id, "movie"]

    def load_movie(self, movie_file: Path) -> None:
        """Load the movie file and stores it as a Pandas dataframe in a
        private attribute.
        """
        movie_df = pd.read_csv(movie_file, sep=",")
        movie_df = movie_df.set_index("movieID")
        self._movie_df = movie_df


if __name__ == "__main__":
    reviewer_movie_matrix = load_reviewer_movie(Path("user_movie_rating_vf.csv"))
    print(reviewer_movie_matrix)

    # movie_retriever = MovieRetriever()
    # movie_retriever.load_movie(Path("movies.csv"))
    # movie = movie_retriever.get_movie_name_from_id(1)
    # print(movie)
