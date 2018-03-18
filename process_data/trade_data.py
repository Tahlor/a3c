# Data from:
# https://manunalepa.wordpress.com/2017/11/14/bitcoin-ethereum-litecoin-exchanges-raw-data-from-coinbase-gdax-are-available-here/

import numpy as np
from io import StringIO
import csv
import os
import shutil
from utils import *

from datetime import datetime
datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')


class TradeData:
    def __init__(self, dataset_path, rows = float("inf")):
        self.input_path = dataset_path
        self.last_row = rows

        self.load_input() # set self.data

    def load_input(self):
        # input is loaded as part of model initialization
        print("Loading data...")
        self.data = []

        # Open CSV
        if self.input_path[-4:] == ".csv":
            #self.load_csv()

            # Load as numpy
            self.data = np.genfromtxt(self.input_path, names="price, side, amount, time", dtype="float16, byte, float16, float64", delimiter=',',
                                   usecols=(1, 2, 3, 4), unpack=True, skip_header=1,
                                   converters={2:buy_sell_encoder, 4: getDateTimeFromISO8601String})
        # Open a Numpy thing
        else:
            self.data = numpy.load(input_path)
        return self.data

    def load_csv(self):
        with open(self.input_path, "r") as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='"')
            for n, line in enumerate(csv_reader):
                self.data.append(line)
                if n > self.last_row:
                    break
                if n % 100 == 0:
                    print("Loading row {}".format(n))

    def write_out(self, output_path):
        with open(output_path,  'w', newline='') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_NONE)

            for i in self.data:
                wr.writerow(i)

    def save_np(self, output_path):
        np.save(output_path, self.data)

    def generate_prices_at_time(self, seconds = 60):
        current_time = myData.data[0]["time"]
        target = round_to_nearest(current_time, round_by=seconds)
        previous_target = target
        self.prices_at_time = []

        for i in self.data:
            #print(target, i["time"])
            if i["time"] > target:
                target = round_to_nearest(i["time"], seconds)
                time_steps = int((target-previous_target)/60 )
                self.prices_at_time += [None]*time_steps + [i["price"]]
                previous_target = target
                target += 60

                # iterpolate over missing timesteps?
                # do nothing?



def create_small_dataset():
    dataset = r"D:\Data\Crypto\GDAX\BTC-USD.csv"
    new_dataset_name = r"BTC-USD_VERY_SHORT.csv"
    dataset_out = os.path.join(r"D:\Data\Crypto\GDAX", new_dataset_name)
    dataset_small = os.path.join(r"./data", new_dataset_name)

    myData = TradeData(dataset, rows = 1000)
    print(myData.data)
    myData.write_out(dataset_out)
    shutil.copy(dataset_out, dataset_small)

if __name__ == "__main__":
    #create_small_dataset()
    if True:
        dataset_small = r"./data/BTC-USD_VERY_SHORT.csv"
        dataset_small = r"./data/BTC-USD_SHORT.csv"
        dataset_small = r"./data/GDAX/BTC-USD.csv"

        myData = TradeData(dataset_small)
        myData.save_np(dataset_small.replace(".csv", ".npy"))
        print(myData.data)
        print(myData.data[0]["price"])
        myData.generate_prices_at_time()
        print(myData.prices_at_time)