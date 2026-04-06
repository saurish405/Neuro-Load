import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_nasa_logs(file_path):
    """
    Cleans the NASA HTTP log TSV from Kaggle.
    The 'time' column is a Unix timestamp integer (seconds since epoch).
    Some Kaggle versions have it as a formatted string — both are handled.
    """
    print(f"Loading {file_path}...")

    df = pd.read_csv(
        file_path,
        sep='\t',
        encoding='latin1',
        low_memory=False,
        on_bad_lines='skip'
    )

    df.columns = df.columns.str.strip()
    print(f"Columns found: {list(df.columns)}")
    print(f"Rows loaded: {len(df)}")

    # --- FIX 1: Robust timestamp parsing ---
    # The NASA Kaggle TSV stores 'time' as a Unix integer (e.g. 807256800).
    # Some versions store it as a CLF string like "[01/Jul/1995:00:00:01 -0400]".
    if 'time' not in df.columns:
        raise ValueError(f"No 'time' column found. Available columns: {list(df.columns)}")

    time_col = df['time'].astype(str).str.strip()

    parsed = pd.to_numeric(time_col, errors='coerce')
    if parsed.notna().sum() > len(df) * 0.5:
        # Unix timestamp
        df['time'] = pd.to_datetime(parsed, unit='s', errors='coerce')
    else:
        # CLF format: [01/Jul/1995:00:00:01 -0400]
        time_col_clean = time_col.str.strip('[]')
        df['time'] = pd.to_datetime(
            time_col_clean,
            format='%d/%b/%Y:%H:%M:%S %z',
            errors='coerce',
            utc=True
        )

    before = len(df)
    df = df.dropna(subset=['time'])
    print(f"Rows after timestamp parsing: {len(df)} (dropped {before - len(df)} unparseable rows)")

    if len(df) == 0:
        raise ValueError(
            "All rows were dropped during timestamp parsing. "
            "Check your TSV file format and the 'time' column values."
        )

    # --- FIX 2: Clean bytes column ---
    if 'bytes' in df.columns:
        df['bytes'] = pd.to_numeric(df['bytes'], errors='coerce').fillna(0)
    else:
        df['bytes'] = 0

    # --- Aggregation: 1-minute windows ---
    print("Resampling traffic to 1-minute intervals...")
    agg_cols = {'bytes': 'sum'}

    host_col = next((c for c in df.columns if c in ('host', 'client', 'ip', 'request')), None)
    if host_col:
        agg_cols[host_col] = 'count'
        rename_map = {host_col: 'request_count'}
    else:
        df['_count'] = 1
        agg_cols['_count'] = 'sum'
        rename_map = {'_count': 'request_count'}

    traffic_df = (
        df.set_index('time')
        .resample('1min')
        .agg(agg_cols)
        .rename(columns=rename_map)
        .fillna(0)
    )

    print(f"Aggregated to {len(traffic_df)} one-minute buckets.")

    if len(traffic_df) < 70:
        raise ValueError(
            f"Only {len(traffic_df)} minute-buckets after aggregation — not enough to train (need >70). "
            "Check that your TSV contains many rows across multiple minutes."
        )

    # --- FIX 3: Normalise ONCE, consistently ---
    scaler = MinMaxScaler()
    traffic_df[['request_count', 'bytes']] = scaler.fit_transform(
        traffic_df[['request_count', 'bytes']]
    )

    # Target: next minute's request count
    traffic_df['target_load'] = traffic_df['request_count'].shift(-1)
    traffic_df.dropna(inplace=True)

    print(f"Clean dataset ready: {len(traffic_df)} rows.")
    return traffic_df
