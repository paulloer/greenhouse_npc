import pandas as pd
from datetime import datetime, timedelta
from constants import COLUMNS, CSV_PATH

# Set start and end times
start_time = datetime.now() - timedelta(hours=48)
end_time = datetime.now() + timedelta(weeks=2)

# Generate timestamps every 30 seconds
timestamps = pd.date_range(start=start_time, end=end_time, freq='30S')

# Create DataFrame with zeros and timestamps
data = pd.DataFrame(0, index=range(len(timestamps)), columns=COLUMNS[1:])  # all zeros
data.insert(0, "Date", timestamps.strftime("%Y-%m-%d %H:%M:%S"))  # formatted datetime

# Save to CSV
CSV_PATH_DEBUG = CSV_PATH.replace(".csv", "_debug.csv")
data.to_csv(CSV_PATH_DEBUG, index=False)
print("CSV file created successfully.")
