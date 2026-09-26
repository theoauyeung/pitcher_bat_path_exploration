"""
Local DuckDB connection with views over the project parquets.

Usage
-----
    import db
    con = db.connect()
    con.sql("SELECT pitcher_id, AVG(disruption_tax) FROM xrv GROUP BY pitcher_id ORDER BY 2").df()
    con.sql("SELECT COLUMNS('pc150_.*') FROM swings LIMIT 5").df()
    con.sql("SELECT * FROM swings JOIN xrv USING (game_pk, at_bat_number, pitch_number) LIMIT 10").df()

Views
-----
    swings    — data/swings_precommit.parquet    (773k rows, 154 cols)
    xrv       — results/xrv_causal.parquet       (per-swing causal decomposition)
    swing_xrv — results/swing_xrv.parquet        (pitch + swing shape + outcome features)
    intended  — models/intended_df.parquet        (Phase A per-swing intended swing shape)
"""

import duckdb

_VIEWS = {
    "swings":    "data/swings_precommit.parquet",
    "xrv":       "results/xrv_causal.parquet",
    "swing_xrv": "results/swing_xrv.parquet",
    "intended":  "models/intended_df.parquet",
}


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for name, path in _VIEWS.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    return con
