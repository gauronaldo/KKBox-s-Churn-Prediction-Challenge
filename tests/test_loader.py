from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_members, load_train


def test_load_train_validates_binary_target(tmp_path: Path) -> None:
    raw_dir = tmp_path
    pd.DataFrame({"msno": ["u1", "u2"], "is_churn": [0, 2]}).to_csv(
        raw_dir / "train.csv", index=False
    )

    with pytest.raises(ValueError, match="0/1"):
        load_train(raw_dir)


def test_load_members_parses_registration_date(tmp_path: Path) -> None:
    raw_dir = tmp_path
    pd.DataFrame(
        {
            "msno": ["u1"],
            "city": [1],
            "bd": [25],
            "gender": ["male"],
            "registered_via": [7],
            "registration_init_time": ["20160101"],
        }
    ).to_csv(raw_dir / "members_v3.csv", index=False)

    members = load_members(raw_dir)

    assert pd.api.types.is_datetime64_any_dtype(members["registration_init_time"])
    assert members.loc[0, "registration_init_time"] == pd.Timestamp("2016-01-01")
