from pyOpenBCI import OpenBCICyton
import csv
import time
import os.path

import time

start_time = time.time()
time_limit = 15  # 15 seconds

label='light_off'
# define a callback function to print the raw data
def print_raw(sample):
    # create a CSV file to write the data to, or open the file in append mode if it already exists
    if os.path.exists('eeg_data_new.csv'):
        csvfile = open('eeg_data_new.csv', 'a')
        writer = csv.writer(csvfile)
    else:
        csvfile = open('eeg_data_new.csv', 'w')
        writer = csv.writer(csvfile)
        # write the column headers for the 8 channels and the timestamp
        writer.writerow(["timestamp", "channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8", "label"])

    # get the current timestamp in milliseconds
    timestamp = int(time.time() * 1000)

    # write the timestamp and channel data for each sample to the CSV file
    writer.writerow([timestamp] + sample.channels_data + [label])
    csvfile.close()
    print(sample.channels_data)

# create an OpenBCI object and connect to the Cyton board
board = OpenBCICyton(port='COM4', daisy=False)

# set the sample rate for the board
#board.set_sample_rate(250)
# start the stream and pass the callback function
board.start_stream(print_raw)

# collect data 15 seconds long
while True:
    time.sleep(1)
    # Check if the time limit has been reached
    if time.time() - start_time > time_limit:
        break
