from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# def preprocessing_for_logreg(extracted_features):
#     # Identify categorical and numerical columns
#     categorical_cols = extracted_features.select_dtypes(
#         include=["object", "category"]
#     ).columns
#     numerical_cols = extracted_features.select_dtypes(include=["number"]).columns
#
#     # Define transformers for categorical and numerical columns
#     categorical_transformer = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
#             ("onehot", OneHotEncoder(handle_unknown="ignore")),
#         ]
#     )
#
#     numerical_transformer = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="mean")),
#             ("scaler", StandardScaler()),
#         ]
#     )
#
#     # Combine transformers into a ColumnTransformer
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numerical_transformer, numerical_cols),
#             ("cat", categorical_transformer, categorical_cols),
#         ]
#     )
#
#     return preprocessor
def preprocessing_for_logreg():
    # Define transformers for categorical and numerical columns
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Create a ColumnTransformer with dynamic column selection
    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, make_column_selector(dtype_include='number')),
        ("cat", categorical_transformer, make_column_selector(dtype_include=["object", "category"]))
    ])

    return preprocessor
