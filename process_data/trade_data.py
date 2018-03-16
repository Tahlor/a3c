# Data from:
# https://manunalepa.wordpress.com/2017/11/14/bitcoin-ethereum-litecoin-exchanges-raw-data-from-coinbase-gdax-are-available-here/

import numpy as np
from io import StringIO
import csv
import os
import shutil
from utils import getDateTimeFromISO8601String

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
            # ndmin=2
            self.data = np.loadtxt(self.input_path, dtype='float16, str, float16, str', delimiter=',', usecols=(1,2,3,4), unpack=True, skiprows = 1, converters = {4: getDateTimeFromISO8601String})

        # Open a Numpy thing
        else:
            pass
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

    def generate_prices_at_time(self, seconds = 60):
        pass

def create_small_dataset():
    dataset = "D:\Data\Crypto\GDAX\BTC-USD.csv"
    new_dataset_name = "BTC-USD_VERY_SHORT.csv"
    dataset_out = os.path.join(r"D:\Data\Crypto\GDAX", new_dataset_name)
    dataset_small = os.path.join(r"./data", new_dataset_name)

    myData = TradeData(dataset, rows = 1000)
    print(myData.data)
    myData.write_out(dataset_out)
    shutil.copy(dataset_out, dataset_small)

if __name__ == "__main__":
    #create_small_dataset()
    if True:
        dataset_small = "./data/BTC-USD_VERY_SHORT.csv"
        myData = TradeData(dataset_small)
        print(myData.data)