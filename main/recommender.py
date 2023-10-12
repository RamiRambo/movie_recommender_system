"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""

from pathlib import Path
from typing import Tuple, List
import numpy as np

import implicit
import scipy

from main.data import load_reviewer_movie, MovieRetriever
from sklearn.metrics import mean_squared_error


class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - movie_retriever: a MovieRetriever instance
        - implicit_model: an implicit model
    """

    def __init__(
        self,
        movie_retriever: MovieRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.movie_retriever = movie_retriever
        self.implicit_model = implicit_model

    def fit(self, user_movie_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user movie matrix."""
        self.implicit_model.fit(user_movie_matrix)

    def recommend(
        self,
        userID: int,
        user_movie_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        movie_ids, ratings = self.implicit_model.recommend(
            userID, user_movie_matrix[n], N=n
        )
        movie = [
            self.movie_retriever.get_movie_name_from_id(movieID)
            for movieID in movie_ids
        ]
        return movie, ratings


if __name__ == "__main__":
    # load reviewer movie matrix
    user_movie = load_reviewer_movie(Path("user_movie_rating_vf.csv"))

    # instantiate movie retriever
    movie_retriever = MovieRetriever()
    movie_retriever.load_movie(Path("movies.csv"))

    # instantiate ALS using implicit
    implict_model = implicit.als.AlternatingLeastSquares(
        factors=10, iterations=10, regularization=0.01
    )

    # instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(movie_retriever, implict_model)
    recommender.fit(user_movie)
    movie, predicted_ratings = recommender.recommend(5, user_movie, n=5)

    # load the actual user-movie ratings
    # actual_ratings = user_movie.toarray()  # Convert user-movie matrix to a dense array

    # Calculate Mean Squared Error (MSE)
    # mse = mean_squared_error(actual_ratings[5], predicted_ratings)

    # Calculate Root Mean Squared Error (RMSE)
    # rmse = np.sqrt(mse)

    # print(f"Mean Squared Error (MSE): {mse}")
    # print(f"Root Mean Squared Error (RMSE): {rmse}")

    # print results
    for movie, rating in zip(movie, predicted_ratings):
        print(f"{movie}: {rating}")
