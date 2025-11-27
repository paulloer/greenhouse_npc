import multiprocessing
from multiprocessing import Process, Lock
from multiprocessing.managers import ListProxy
from datetime import datetime, timedelta
import pandas as pd
import os
import csv
import random

# from sensor import measurement_loop
from controller import controller_loop
from online_opt import online_opt
from online_learning import online_learning
from periodic_saver import periodic_saver
from generate_random_targets import generate_random_targets
from constants import CSV_PATH, ARGUMENTS, STAND_VALUES
CSV_PATH_DEBUG = CSV_PATH.replace(".csv", "_debug.csv")



def load_last_hrs_to_shared_list(shared_measurements, hours):
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])

        # for debugging:
        if True:
            now = datetime.now()
            cutoff = now - timedelta(hours=hours) - timedelta(seconds=30)

            # Keep only rows from last 24h
            df_recent = df[df["Date"] >= cutoff].copy()  # <--- add .copy() to ensure modifications work


            # Convert Timestamps to string format
            df_recent["Date"] = df_recent["Date"].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Convert to list of lists and assign
            shared_measurements[:] = df_recent.values.tolist()
        else:
            df["Date"] = df["Date"].dt.strftime('%Y-%m-%d %H:%M:%S')
            recent_measurements = df.values.tolist()
            shared_measurements[:] = recent_measurements[-hours*120:]

        print(f"[✓] Loaded {len(shared_measurements)} rows from last {hours}h into shared_measurements")
    else:
        print("[i] CSV file not found — shared_measurements will start empty.")


def append_debug_data(CSV_PATH):
    """
    Appends 120 debug samples (1 every 30s for the past 60min)
    with fixed greenhouse values to the CSV at CSV_PATH.
    """
    # Field names (same order as in your example)
    fieldnames = [
        "Date","Temperature_inside","Humidity_inside","Temperature_outside","Humidity_outside",
        "Radiation_inside","Radiation_outside","Wind_speed_outside",
        "Vent_S1_Roof_1","Vent_S1_Roof_2","Vent_S1_Roof_3",
        "Vent_S1_Side_NW","Vent_S1_Side_S","Vent_S1_Side_N","Vent_S1_Side_SW",
        "Vent_S2_Roof_1","Vent_S2_Roof_2","Vent_S2_Roof_3",
        "Vent_S2_Side_E","Vent_S2_Side_S","Vent_S2_Side_N"
    ]

    # Base values
    temp_inside = 20
    temp_outside = 20
    humidity_inside = 70
    humidity_outside = 70
    wind_speed = 0
    radiation_inside = 0
    radiation_outside = 0
    vent_value = 100

    # Generate 120 samples (every 30 seconds)
    now = datetime.now()
    start_time = now - timedelta(minutes=60)
    timestamps = [start_time + timedelta(seconds=30*i) for i in range(120)]

    # Open CSV for appending
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        for ts in timestamps:
            row = [
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                round(temp_inside + random.uniform(-0.2, 0.2), 2),
                round(humidity_inside + random.uniform(-0.5, 0.5), 2),
                round(temp_outside + random.uniform(-0.2, 0.2), 2),
                round(humidity_outside + random.uniform(-0.5, 0.5), 2),
                radiation_inside,
                radiation_outside,
                wind_speed,
                vent_value, vent_value, vent_value,
                vent_value, vent_value, vent_value, vent_value,
                vent_value, vent_value, vent_value,
                vent_value, vent_value, vent_value
            ]
            writer.writerow(row)



if __name__ == '__main__':
    manager =  multiprocessing.Manager()
    shared_measurements = manager.list()  # Safe shared memory
    shared_control_history = manager.list()  # Safe shared memory
    shared_vent_postion_targets = manager.list()  # Safe shared memory
    shared_model = manager.dict()
    lock_m = Lock()
    lock_c = Lock()
    lock_v = Lock()
    lock_p = Lock()

    # append_debug_data(CSV_PATH)

    # initiliaze shared measurements with the last hours of data, if available
    load_last_hrs_to_shared_list(shared_measurements, 1)

    # initialize with last measured control, as this is the basis for optimization
    try:
        shared_vent_postion_targets.append(shared_measurements[-1][11])
    except IndexError:
        shared_vent_postion_targets.append(0)

    shared_model['path'] = ARGUMENTS.model_path
    shared_model['mean_values'] = STAND_VALUES.mean_values
    shared_model['std_values'] = STAND_VALUES.std_values

    processes = []

    # controller loop compares actuals and targets of the vents every 30 seconds and takes a measurement of all states
    processes.append(Process(target=controller_loop, args=(shared_measurements, shared_vent_postion_targets, lock_m, lock_v)))
    # online optimization needs samples every 30 seconds and updates the vent position targetes every 30 seconds
    processes.append(Process(target=online_opt, args=(shared_measurements, shared_vent_postion_targets, shared_model, lock_m, lock_v, lock_p)))
    # online learning uses collected data from csv to train a new model on the new data
    # processes.append(Process(target=online_learning, args=(shared_model, lock_p)))
    # save measurements every hour to csv
    processes.append(Process(target=periodic_saver, args=(shared_measurements, lock_m)))
    # generate random targets for the vent position during dataset creation
    # processes.append(Process(target=generate_random_targets, args=(shared_vent_postion_targets, lock_v, 180)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()
        