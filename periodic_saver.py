import os
import pandas as pd
import time
from datetime import datetime, timedelta

from constants import CSV_PATH, COLUMNS

import pandas as pd
import os
import time
from datetime import datetime
from threading import Lock

# Global variable to store the last saved timestamp (as datetime object)
last_saved_timestamp = None
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"  # adjust if your format differs

def save_shared_measurements_to_csv(shared_measurements, lock_m):
    global last_saved_timestamp

    with lock_m:
        # Copy to avoid holding the lock too long
        data_copy = list(shared_measurements)

    new_entries = []

    for row in data_copy:
        try:
            timestamp_str = row[0]
            timestamp = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
        except Exception as e:
            print(f"Skipping row with invalid timestamp: {row[0]} ({e})")
            continue

        if last_saved_timestamp is None or timestamp > last_saved_timestamp:
            new_entries.append((timestamp, row))

    if not new_entries:
        return

    # Sort new entries by timestamp
    new_entries.sort(key=lambda x: x[0])
    last_saved_timestamp = new_entries[-1][0]

    # Drop parsed timestamps, keep raw rows
    new_rows = [row for _, row in new_entries]

    df = pd.DataFrame(new_rows, columns=COLUMNS)

    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_PATH, mode='w', header=True, index=False)


def init_last_saved_timestamp():
    global last_saved_timestamp
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            if not df.empty:
                ts_str = df.iloc[-1, 0]  # assumes timestamp is first column
                last_saved_timestamp = datetime.strptime(ts_str, TIMESTAMP_FORMAT)
        except Exception as e:
            print(f"Could not load last timestamp from CSV: {e}")


def periodic_saver(shared_measurements, lock_m):
    init_last_saved_timestamp()
    while True:
        save_shared_measurements_to_csv(shared_measurements, lock_m)
        time.sleep(30)  # wait x seconds

