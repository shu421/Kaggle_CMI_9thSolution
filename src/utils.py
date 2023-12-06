import pdb

import numpy as np
import polars as pl
from scipy.signal import find_peaks

from .metrics import event_detection_ap


def get_event(input_df: pl.DataFrame, thr: float = 0.1, size: int = 12 * 10) -> pl.DataFrame:
    output_df = input_df

    # 極大値を取得
    onset_id = output_df.select("step").to_numpy().flatten()
    onset_pred = output_df["onset"]
    onset_index = find_peaks(onset_pred, height=thr, distance=size)[0]
    onset_id = onset_id[onset_index]

    wakeup_id = output_df.select("step").to_numpy().flatten()
    wakeup_pred = output_df["wakeup"]
    wakeup_index = find_peaks(wakeup_pred, height=thr, distance=size)[0]
    wakeup_id = wakeup_id[wakeup_index]

    # 極大値が存在したら評価値を計算
    if (len(onset_id) > 0) and (len(wakeup_id) > 0):
        # Ensuring all predicted sleep periods begin and end
        if min(wakeup_id) < min(onset_id):
            wakeup_id = wakeup_id[1:]
            if len(wakeup_id) == 0:
                output_df = pl.DataFrame(
                    schema={
                        "series_id": str,
                        "step": pl.UInt32,
                        "event": str,
                        "onset": pl.Float32,
                        "wakeup": pl.Float32,
                    }
                )
                return output_df

        if max(onset_id) > max(wakeup_id):
            onset_id = onset_id[:-1]
            if len(onset_id) == 0:
                output_df = pl.DataFrame(
                    schema={
                        "series_id": str,
                        "step": pl.UInt32,
                        "event": str,
                        "onset": pl.Float32,
                        "wakeup": pl.Float32,
                    }
                )
                return output_df

        # イベント列の作成のためのマスクを計算
        mask_onset = pl.col("step").is_in(onset_id)
        mask_wakeup = pl.col("step").is_in(wakeup_id)

        event_expr = (
            pl.when(mask_onset)
            .then("onset")
            .when(mask_wakeup)
            .then("wakeup")
            .otherwise(None)
            .alias("event")
        )
        output_df = output_df.with_columns(event_expr)

        # 必要な行の選択
        idx = np.concatenate([onset_id, wakeup_id])
        idx = np.sort(idx)
        output_df = output_df.filter(pl.col("step").is_in(idx))
    else:
        output_df = pl.DataFrame(
            schema={
                "series_id": str,
                "step": pl.UInt32,
                "event": str,
                "onset": pl.Float32,
                "wakeup": pl.Float32,
            }
        )

    output_df = output_df.select(["series_id", "step", "event", "onset", "wakeup"])
    return output_df


def get_score(
    true_df: pl.DataFrame,
    pred_df: pl.DataFrame,
    preds: np.array,
    thr: float = 0.1,
    size: int = 12 * 60 * 6,
):
    """
    Input
    -----
    true_df: pl.DataFrame, events
        columns: series_id, timestamp, asleep
    pred_df: pl.DataFrame, series
        columns: series_id, timestamp, asleep
    preds: np.ndarray
    """

    true_df = true_df.filter(pl.col("step").is_not_null())
    true_df = true_df.select(["series_id", "step", "event", "onset", "wakeup"])

    preds = preds.reshape(-1, 2)
    pred_df = pred_df.with_columns(
        (pl.lit(preds[:, 0]).cast(pl.Float32).alias("onset")),
        (pl.lit(preds[:, 1]).cast(pl.Float32).alias("wakeup")),
    )

    # sliding window predictionを実施している場合は以下の処理が必要
    pred_df = pred_df.group_by(["series_id", "step"], maintain_order=True).agg(
        (pl.col("onset").mean().alias("onset")),
        (pl.col("wakeup").mean().alias("wakeup")),
    )

    # series_idごとにイベントを取得
    pred_df = pred_df.group_by("series_id", maintain_order=True).map_groups(
        lambda x: get_event(x, thr=thr, size=size)
    )

    # score列の作成
    pred_df = pred_df.with_columns(
        pl.when(pl.col("event") == "onset")
        .then(pl.col("onset"))
        .otherwise(pl.col("wakeup"))
        .alias("score")
    )

    score = event_detection_ap(true_df.to_pandas(), pred_df.to_pandas())
    return score

