# Imports
import joblib as jb
from scipy.sparse import hstack, csr_matrix
import numpy as np

# Loads machine learning models
logistic_regression_model = jb.load("Logistic_reg_animes.pkl.z")
lightgbm_model = jb.load("lgbm_animes.pkl.z")
word_vectorized = jb.load("word_vectorizer_animes.pkl.z")


def compute_features(data):
    """The function to vectorize the title and organize all features for the models."""

    if data is None:
        return None

    title = data['title']

    features = dict()
    features['mean_score'] = data['mean_score']
    features['num_episodes'] = data['num_episodes']

    vectorized_title = word_vectorized.transform([title])

    num_features = csr_matrix(np.array([features['mean_score'], features['num_episodes']]))
    feature_array = hstack([num_features, vectorized_title])

    return feature_array


def compute_prediction(data):
    """Apply machine leaning models and returns a score for each feature in data."""

    feature_array = compute_features(data)

    if feature_array is None:
        return 0

    probability_score_logistic = logistic_regression_model.predict_proba(feature_array)[0][1]
    probability_score_lgbm = lightgbm_model.predict_proba(feature_array)[0][1]
    probability_score = 0.3 * probability_score_logistic + 0.7 * probability_score_lgbm

    return probability_score
