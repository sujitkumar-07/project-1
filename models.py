from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from typing import Tuple


def build_linear_regression() -> Tuple[str, object]:
    model = make_pipeline(StandardScaler(), LinearRegression())
    return 'linear', model


def build_random_forest(n_estimators: int = 100, random_state: int = 42) -> Tuple[str, object]:
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
    return 'rf', model


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
