from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn import decomposition


@dataclass
class ModelTrainingResult:
    # model: BaseModel

    y_test: pd.DataFrame
    x_test: pd.DataFrame
    predictions: pd.DataFrame
    probabilities: pd.DataFrame
    probabilities_match_id: pd.DataFrame

    metrics: dict
    class_accuracies: dict
    feature_importances: Optional[pd.DataFrame] = None


def get_pca_explained(df):
    component_n = min(10, len(df.columns) - 1)

    _pca_11_labels = [f"PC{i + 1}" for i in range(component_n)]

    _pca_11 = decomposition.PCA(n_components=component_n)
    _pca_11.fit(df)

    pca_transformed = _pca_11.transform(df)

    pca_df = pd.DataFrame(data=pca_transformed, index=df.index, columns=_pca_11_labels)
    # pca_df[target_var] = result_no_nan

    _explained_pc = pd.DataFrame(
        {"var": _pca_11.explained_variance_ratio_, "PC": _pca_11_labels}
    )
    _explained_pc["cum_var"] = _explained_pc["var"].cumsum()
    return _explained_pc
