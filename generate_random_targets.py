import time
from datetime import datetime, timedelta
from opc_client import gh_client
from constants import TIME_CONTROL, URL, OPC_CONTROL_PARAM_DICT, OPC_PARAMETERS
import numpy as np

# control loop with encoders
def generate_random_targets(shared_vent_postion_targets, lock_v, time_sleep=300):

    while True:
        current_time = datetime.now()
        if 10 <= current_time.hour < 17: # TODO generate data during night too
            last_position = shared_vent_postion_targets[-1]
            new_position = np.random.uniform(0, 100)
            if last_position-10 <= new_position <= last_position+10:
                r = np.random.randint(0,2)
                new_position = last_position + r * 10 - (1-r) * 10
            new_position = np.max([0, np.min([new_position, 100])])
            # new_position if new_position < last_position - 10 else new_position + 20 # at least 10 % difference to old position
            print(f'Next target control is {new_position}')
            with lock_v:
                shared_vent_postion_targets.append(new_position)
        else:
            new_position = 100
            print(f'Next target control is {new_position:.2f}')
            with lock_v:
                shared_vent_postion_targets.append(new_position)

        time.sleep(time_sleep)  