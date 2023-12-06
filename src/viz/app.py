import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import streamlit as st

sys.path.append("../../")
from src.utils import get_score

# st.set_page_config(layout="wide")
plt.rcParams["figure.figsize"] = (20, 5)
plt.style.use("ggplot")


class Config:
    AUTHOR = "shu421"

    EXP = ""
    BASE_PATH = "/home/working/"
    API_PATH = "/root/.kaggle/kaggle.json"
    COMPETITION = "child-mind-institute-detect-sleep-states"
    INPUT_PATH = Path(BASE_PATH) / "input"
    OUTPUT_PATH = Path(BASE_PATH) / "output"

    target_cols = ["asleep", "onset", "wakeup"]
    n_targets = len(target_cols)

    seed = 42

    # metrics
    thr = 0.1
    size = 12 * 10  # 10 min


def hash_polars_column_factory(obj):
    # ここでは単純なハッシュ関数を使用しますが、
    # 必要に応じてより複雑なロジックを実装できます。
    return str(obj)

def hash_polars_expr(obj):
    # ここでは単純なハッシュ関数を使用しますが、
    # 必要に応じてより複雑なロジックを実装できます。
    return str(obj)


# Function to load data
def load_data():
    train_series_df = pl.read_parquet(cfg.INPUT_PATH / "train_series.parquet")
    train_events_df = pl.read_csv(cfg.INPUT_PATH / "train_events.csv")

    # preprocessing
    train_events_df = train_events_df.drop_nulls()
    train_events_df = train_events_df.filter(
        ~(
            (
                (train_events_df["series_id"] == "0ce74d6d2106")
                & (train_events_df["night"] == 20)
            )
            | (
                (train_events_df["series_id"] == "154fe824ed87")
                & (train_events_df["night"] == 30)
            )
            | (
                (train_events_df["series_id"] == "44a41bba1ee7")
                & (train_events_df["night"] == 10)
            )
            | (
                (train_events_df["series_id"] == "efbfc4526d58")
                & (train_events_df["night"] == 7)
            )
            | (
                (train_events_df["series_id"] == "f8a8da8bdd00")
                & (train_events_df["night"] == 17)
            )
        )
    )

    # create target column
    train_events_df = train_events_df.with_columns(
        pl.when(pl.col("event") == "onset").then(0).otherwise(1).alias("asleep"),
        pl.when(pl.col("event") == "onset").then(1).otherwise(0).alias("onset"),
        pl.when(pl.col("event") == "wakeup").then(1).otherwise(0).alias("wakeup"),
    )
    train_series_df = train_series_df.join(
        train_events_df.select(["series_id", "timestamp", "night"] + cfg.target_cols),
        on=["series_id", "timestamp"],
        how="left",
    )

    # series_idごとに17280ごとに番号を割り当てる
    segments = []
    for series, group in train_series_df.group_by("series_id", maintain_order=True):
        group_id = 0
        start = 0
        while start < len(group):
            end = start + 17280
            _df = group[start:end]
            _df = _df.with_columns(pl.lit(group_id).alias("group"))
            segments.append(_df)
            start = start + 17280
            group_id += 1

    train_series_df = pl.concat(segments)

    train_series_df = train_series_df.with_columns(
        (pl.col("asleep").backward_fill().fill_null(0)),
        (pl.col("onset").fill_null(0)),
        (pl.col("wakeup").fill_null(0)),
    )

    return train_series_df, train_events_df


# @st.cache
def plot_all_sequence(train_series_df, series_id):
    """plot all data for each series"""
    tmp = train_series_df.filter(pl.col("series_id") == series_id)

    fig, ax = plt.subplots()
    sns.lineplot(x="step", y="anglez", data=tmp, color="orange", alpha=0.5, label="anglez")

    ax2 = ax.twinx()
    ax2.set_yticks(np.arange(0, 1.1, 0.2))
    sns.lineplot(x="step", y="asleep", data=tmp, ax=ax2, color="darkred", label="asleep", alpha=0.8)
    sns.lineplot(x="step", y="onset_pred", data=tmp, ax=ax2, color="darkblue", label="onset", alpha=0.8)
    sns.lineplot(x="step", y="wakeup_pred", data=tmp, ax=ax2, color="purple", label="wakeup", alpha=0.8)

    ax.legend(loc="upper left", bbox_to_anchor=(0, 1))
    ax2.legend(loc="upper right", bbox_to_anchor=(1, 1))

    ax.set_title(f"series_id: {series_id}")

    st.pyplot(fig)


# @st.cache
def plot_pred_series(train_series_df, series_id, start_group, end_group):
    """plot predicted data for each group"""
    for group in range(start_group, end_group+1):
        tmp = train_series_df.filter(
            (pl.col("series_id") == series_id)
            & (pl.col("group")==group)
        )

        fig, ax = plt.subplots()
        sns.lineplot(x="step", y="anglez", data=tmp, color="orange", alpha=0.5, label="anglez")

        ax2 = ax.twinx()
        ax2.set_yticks(np.arange(0, 1.1, 0.2))
        sns.lineplot(x="step", y="asleep", data=tmp, ax=ax2, color="darkred", label="asleep", alpha=0.8)
        sns.lineplot(x="step", y="onset_pred", data=tmp, ax=ax2, color="darkblue", label="onset", alpha=0.8)
        sns.lineplot(x="step", y="wakeup_pred", data=tmp, ax=ax2, color="purple", label="wakeup", alpha=0.8)

        ax.legend(loc="upper left", bbox_to_anchor=(0, 1))
        ax2.legend(loc="upper right", bbox_to_anchor=(1, 1))

        ax.set_title(f"series_id: {series_id}, group: {group}")

        st.pyplot(fig)


def get_score_by_series_id(events_df, series_df, preds, series_id, group=None):
    events_df = events_df.filter(pl.col("step").is_not_null())
    events_df = events_df.select(["series_id", "step", "event", "onset", "wakeup"])

    preds = preds.reshape(-1, 2)
    series_df = series_df.with_columns(
        (pl.lit(preds[:, 0]).cast(pl.Float32).alias("onset")),
        (pl.lit(preds[:, 1]).cast(pl.Float32).alias("wakeup")),
    )

    series_df = series_df.filter(pl.col("series_id") == series_id)
    events_df = events_df.filter(pl.col("series_id") == series_id)
    if group is not None:
        series_df = series_df.filter(pl.col("group") == group)
        events_df = events_df.filter(pl.col("step").is_between(group * 17280, (group + 1) * 17280))
    preds = series_df.select(["onset", "wakeup"]).to_numpy().flatten()

    return get_score(events_df, series_df, preds, thr=cfg.thr, size=cfg.size)


# Streamlit app
def main():
    st.title("Data Visualization")

    # data load
    oof_exp = st.text_input("Experiment ID", "exp072")
    oof = pl.read_parquet(cfg.OUTPUT_PATH / oof_exp / "preds" / "oof.parquet")
    train_series_df, train_events_df = load_data()
    train_series_df = train_series_df.with_columns(
        pl.lit(oof["onset"]).alias("onset_pred"),
        pl.lit(oof["wakeup"]).alias("wakeup_pred"),
    )

    # User inputs
    series_id = st.selectbox("Series id", train_series_df["series_id"].unique(maintain_order=True).to_list())
    start_group = st.number_input("Start Group", min_value=1, max_value=100, value=1)
    end_group = st.number_input("End Group", min_value=2, max_value=100, value=10)

    # Visualization
    if st.button("Plot Predicted Data"):
        plot_pred_series(train_series_df, series_id, start_group, end_group)

    if st.button("Plot All Data"):
        plot_all_sequence(train_series_df, series_id)

    if st.button("Score"):
        score = get_score_by_series_id(
            train_events_df,
            train_series_df,
            oof.select(["onset", "wakeup"]).to_numpy().flatten(),
            series_id,
            group=5,
            )
        st.write(f"series_id: {series_id}, score: {score: .5f}")

if __name__ == "__main__":
    cfg = Config
    # load_data.clear()
    # st.runtime.legacy_caching.clear_cache()
    main()
