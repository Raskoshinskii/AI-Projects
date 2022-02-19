import numpy as np 
from sklearn.metrics import r2_score


def mean_absolute_percentage_error(y_true, y_pred, **kwargs): 
    """
    Calculates adjusted MAPE metric

    y_true: pd.Series or np.array
        True values 
    y_pred: np.array
        Predicted values 
    Returns:
    -------
    int
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def adjusted_rsquare(y_true, y_pred, X, **kwargs):
    """
    Calculates adjusted R2

    y_true: pd.Series or np.array
        True values 
    y_pred: np.array
        Predicted values 
    X: pd.DataFrame or np.array
        Matrix of features     
    Returns:
    -------
    int
    """
    n = y_true.shape[0] # number of observation
    p = X.shape[1] # number of features
    r_squared = r2_score(y_true, y_pred)
    adjusted_r_squared = 1 - (((1 - r_squared)*(n - 1))/(n - p - 1))
    return adjusted_r_squared