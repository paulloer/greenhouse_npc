import time
from datetime import datetime, timedelta
from opc_client import gh_client, URL, OPC_PARAMETERS
from constants import TIME_SENSOR


def measurement_loop(shared_measurements, lock_m):

    # OPC setup
    gh13_client = gh_client(URL)

    # Initial setup
    last_read_time = datetime.min
    
    while True:
        current_time = datetime.now()
        
        # Check if forecast update is needed
        if current_time.second in [0, 30] and (current_time - last_read_time).total_seconds() >= TIME_SENSOR:
            last_read_time = current_time
            measurements = gh13_client.read(OPC_PARAMETERS)
            with lock_m:
                shared_measurements.append([current_time.strftime('%Y-%m-%d %H:%M:%S')] + list(measurements.values()))

        time.sleep(1)