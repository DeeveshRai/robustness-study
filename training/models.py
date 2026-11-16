"""Built-in tabular regressors. Each call constructs an unfitted model."""

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge


MODEL_TYPES = {
    "ridge": Ridge,
    "linear": LinearRegression,
    "elastic_net": ElasticNet,
    "random_forest": RandomForestRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
}


def create_model(name, parameters=None):
    if name not in MODEL_TYPES:
        raise ValueError(f"Unknown model {name!r}; choose from {list(MODEL_TYPES)}")
    parameters = dict(parameters or {})
    if name in ("elastic_net", "random_forest", "hist_gradient_boosting"):
        parameters.setdefault("random_state", 42)
    return MODEL_TYPES[name](**parameters)
