"""Feature engineering and the classifier, bundled into a single sklearn Pipeline.

Keeping both in one artifact guarantees training and inference transform data the same way.
"""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

NUMERIC = ["mean_score", "num_episodes", "start_year", "popularity"]
FEATURES = ["title", "synopsis", "genres", "studios", "media_type", *NUMERIC]

# "Action|Slice of Life" -> ["Action", "Slice of Life"]
PIPE_SEPARATED = r"[^|]+"


def build_pipeline() -> Pipeline:
    features = ColumnTransformer(
        [
            ("title", TfidfVectorizer(ngram_range=(1, 2)), "title"),
            ("synopsis", TfidfVectorizer(stop_words="english", max_features=5000), "synopsis"),
            ("genres", CountVectorizer(token_pattern=PIPE_SEPARATED, binary=True, lowercase=False), "genres"),
            (
                "studios",
                CountVectorizer(token_pattern=PIPE_SEPARATED, binary=True, lowercase=False),
                "studios",
            ),
            ("media_type", OneHotEncoder(handle_unknown="ignore"), ["media_type"]),
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), MaxAbsScaler()), NUMERIC),
        ]
    )
    # Same 3/7 LogReg/LightGBM blend as the original model
    ensemble = VotingClassifier(
        [
            ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000)),
            ("lgbm", LGBMClassifier(class_weight="balanced", random_state=0, verbose=-1)),
        ],
        voting="soft",
        weights=[0.3, 0.7],
    )
    return Pipeline([("features", features), ("model", ensemble)])


def label(user_list: pd.DataFrame, like_threshold: int) -> pd.Series:
    """1 = liked, 0 = disliked or dropped, NaN = no opinion yet (unscored, not dropped)."""
    scored = user_list["list_score"] > 0
    liked = scored & (user_list["list_score"] >= like_threshold)
    disliked = (scored & ~liked) | (user_list["list_status"] == "dropped")
    y = pd.Series(float("nan"), index=user_list.index)
    y[disliked] = 0
    y[liked] = 1
    return y


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """Mean average precision over a stratified 5-fold cross-validation."""
    folds = StratifiedKFold(n_splits=min(5, int(y.value_counts().min())), shuffle=True, random_state=42)
    return cross_val_score(pipeline, X[FEATURES], y, cv=folds, scoring="average_precision").mean()
