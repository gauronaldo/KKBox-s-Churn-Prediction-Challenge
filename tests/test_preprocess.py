import pandas as pd

from src.features.preprocess import apply_preprocessor, build_preprocessor, identify_column_groups


def test_preprocessor_drops_ids_and_transforms_features() -> None:
    frame = pd.DataFrame(
        {
            "msno": ["u1", "u2", "u3"],
            "is_churn": [0, 1, 0],
            "analysis_reference_date": [pd.Timestamp("2017-01-31")] * 3,
            "numeric_feature": [1.0, None, 3.0],
            "category_feature": ["a", "b", None],
        }
    )

    groups = identify_column_groups(frame)
    preprocessor = build_preprocessor(groups)
    preprocessor.fit(frame)
    transformed = apply_preprocessor(preprocessor, frame, "test")

    assert "msno" not in transformed.columns
    assert "is_churn" not in transformed.columns
    assert transformed.shape[0] == len(frame)
    assert transformed.isna().sum().sum() == 0
