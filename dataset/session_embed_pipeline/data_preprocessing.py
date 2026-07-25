import logging
import os

import polars as pl


INPUT_PATH = (
    "/home/snarayana_umass_edu/E4SRec-1/"
    "userid-timestamp-artid-artname-traid-traname.tsv"
)
PARQUET_PATH = "/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/lastfm1k.parquet"

MIN_TRACKS_IN_SESSION = 10
MIN_SESSIONS_IN_HISTORY = 50
SESSION_GAP_SECONDS = 60 * 20
MIN_USER_NSTREAMS = 1000
MIN_ITEM_NSTREAMS = 7


def get_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(__name__)


def build_output_path() -> str:
    input_dir = os.path.dirname(os.path.abspath(__file__))
    filename = (
        "user_sessions_lastfm1k_"
        f"minuser{MIN_USER_NSTREAMS}_"
        f"minitem{MIN_ITEM_NSTREAMS}_"
        f"sessgap{SESSION_GAP_SECONDS}_"
        f"minsesslen{MIN_TRACKS_IN_SESSION}_"
        f"minhist{MIN_SESSIONS_IN_HISTORY}.csv"
    )
    return os.path.join(input_dir, filename)


def ensure_parquet(logger: logging.Logger) -> None:
    if os.path.exists(PARQUET_PATH):
        logger.info("Parquet cache exists at %s", PARQUET_PATH)
        return

    logger.info("Convert TSV to Parquet (quote_char=None, ignore_errors=True)")
    df = pl.read_csv(
        INPUT_PATH,
        separator="\t",
        quote_char=None,
        ignore_errors=True,
        has_header=False,
        new_columns=["userid", "timestamp", "artid", "artname", "traid", "traname"],
    )
    logger.info("Loaded %d rows, writing Parquet to %s", df.height, PARQUET_PATH)
    df.write_parquet(PARQUET_PATH)
    del df


def preprocess_lastfm_1k(logger: logging.Logger) -> pl.DataFrame:
    logger.info("Load data from %s", PARQUET_PATH)
    data = pl.read_parquet(PARQUET_PATH)

    logger.info("Parse timestamps (strip trailing Z)")
    data = (
        data
        .with_columns(
            pl.col("timestamp")
            .str.replace("Z$", "")
            .str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False)
            .alias("ts")
        )
        .filter(pl.col("ts").is_not_null())
    )

    logger.info("Build fallback IDs for missing artist/track MBIDs")
    data = data.with_columns(
        pl.when(pl.col("artid").is_null())
        .then(
            pl.lit("artistname_")
            + pl.col("artname").hash().cast(pl.Utf8)
        )
        .otherwise(pl.col("artid"))
        .alias("art_id"),
        pl.when(pl.col("traid").is_null())
        .then(
            pl.lit("trackname_")
            + pl.concat_str([pl.col("artname"), pl.col("traname")], separator="||")
            .hash()
            .cast(pl.Utf8)
        )
        .otherwise(pl.col("traid"))
        .alias("track_id"),
        pl.col("userid").alias("user_id"),
    )

    logger.info(
        "Filter by minimum streams (user >= %d, track >= %d)",
        MIN_USER_NSTREAMS,
        MIN_ITEM_NSTREAMS,
    )
    user_counts = (
        data.group_by("user_id").agg(pl.len().alias("n"))
        .filter(pl.col("n") >= MIN_USER_NSTREAMS)
        .select("user_id")
    )
    data = data.join(user_counts, on="user_id")

    track_counts = (
        data.group_by("track_id").agg(pl.len().alias("n"))
        .filter(pl.col("n") >= MIN_ITEM_NSTREAMS)
        .select("track_id")
    )
    data = data.join(track_counts, on="track_id")

    logger.info("Organize data into sessions (gap = %ds)", SESSION_GAP_SECONDS)
    data = data.sort(["user_id", "ts"])
    data = data.with_columns(
        pl.col("ts").diff().over("user_id").dt.total_seconds().alias("delta_t")
    )
    data = data.with_columns(
        pl.when(pl.col("delta_t").is_null() | (pl.col("delta_t") > SESSION_GAP_SECONDS))
        .then(1)
        .otherwise(0)
        .alias("is_new_session")
    )
    data = data.with_columns(
        pl.col("is_new_session").cum_sum().over("user_id").alias("session_id")
    )

    logger.info("Deduplicate tracks within sessions")
    data = (
        data.sort(["user_id", "ts"])
        .unique(subset=["user_id", "art_id", "track_id", "session_id"], keep="first")
    )

    all_sess_counts = data.group_by(["user_id", "session_id"]).agg(pl.len().alias("sess_len"))
    sess_len_series = all_sess_counts["sess_len"]
    logger.info(
        "Session length distribution (before filter): "
        "min=%d, 25th=%d, median=%d, 75th=%d, 90th=%d, max=%d, total_sessions=%d",
        sess_len_series.min(),
        sess_len_series.quantile(0.25),
        sess_len_series.quantile(0.50),
        sess_len_series.quantile(0.75),
        sess_len_series.quantile(0.90),
        sess_len_series.max(),
        all_sess_counts.height,
    )

    logger.info("Filter sessions shorter than %d tracks", MIN_TRACKS_IN_SESSION)
    sess_counts = all_sess_counts.filter(pl.col("sess_len") >= MIN_TRACKS_IN_SESSION).select(["user_id", "session_id"])
    logger.info("Sessions surviving filter: %d / %d", sess_counts.height, all_sess_counts.height)
    data = data.join(sess_counts, on=["user_id", "session_id"])

    logger.info("Truncate sessions to first %d tracks", MIN_TRACKS_IN_SESSION)
    data = (
        data.sort(["user_id", "ts"])
        .with_columns(
            pl.col("ts").rank("ordinal").over(["user_id", "session_id"]).alias("pos")
        )
        .filter(pl.col("pos") <= MIN_TRACKS_IN_SESSION)
    )

    all_user_sess = (
        data.select(["user_id", "session_id"]).unique()
        .group_by("user_id").agg(pl.len().alias("n_sessions"))
    )
    n_sess_series = all_user_sess["n_sessions"]
    logger.info(
        "Sessions-per-user distribution (before filter): "
        "min=%d, 25th=%d, median=%d, 75th=%d, 90th=%d, max=%d, total_users=%d",
        n_sess_series.min(),
        n_sess_series.quantile(0.25),
        n_sess_series.quantile(0.50),
        n_sess_series.quantile(0.75),
        n_sess_series.quantile(0.90),
        n_sess_series.max(),
        all_user_sess.height,
    )

    logger.info("Filter users with fewer than %d sessions", MIN_SESSIONS_IN_HISTORY)
    user_sess_counts = all_user_sess.filter(pl.col("n_sessions") >= MIN_SESSIONS_IN_HISTORY).select("user_id")
    logger.info("Users surviving filter: %d / %d", user_sess_counts.height, all_user_sess.height)
    data = data.join(user_sess_counts, on="user_id")

    return data.select(
        [
            "user_id",
            "ts",
            "art_id",
            "artname",
            "track_id",
            "traname",
            "session_id",
        ]
    ).rename({"artname": "artist_name", "traname": "track_name"})


def main() -> None:
    logger = get_logger()
    output_path = build_output_path()

    ensure_parquet(logger)

    if not os.path.exists(output_path):
        user_sessions = preprocess_lastfm_1k(logger)
        logger.info("Write user sessions to %s", output_path)
        user_sessions.write_csv(output_path, include_header=False)
    else:
        logger.info("Read user sessions from %s", output_path)
        user_sessions = pl.read_csv(
            output_path,
            has_header=False,
            new_columns=[
                "user_id",
                "art_id",
                "track_id",
                "artist_name",
                "track_name",
                "ts",
                "session_id",
            ],
        )

    logger.info("Number of interactions: %d", user_sessions.height)
    logger.info("Number of users: %d", user_sessions["user_id"].n_unique())
    logger.info("Number of tracks: %d", user_sessions["track_id"].n_unique())
    logger.info("Number of artists: %d", user_sessions["art_id"].n_unique())


if __name__ == "__main__":
    main()
