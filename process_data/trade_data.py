# Data from:
# https://manunalepa.wordpress.com/2017/11/14/bitcoin-ethereum-litecoin-exchanges-raw-data-from-coinbase-gdax-are-available-here/

import numpy as np

class TradeData:
    def __init__(self, dataset_path, rows = float("inf")):
        self.input_path = dataset_path
        self.last_row = rows
        self.data = self.load_input()

    def load_input(self):
        # input is loaded as part of model initialization
        print("Loading data...")
        data = []

        # Open CSV
        if self.input_path[-4:] == ".csv":
            with open(self.input_path) as f:
                for n, line in enumerate(f.readlines()):
                    data.append(line.strip().split(","))
                    if n > self.last_row :
                        break
                    if n % 100 == 0:
                        print("Loading row {}".format(n))

        # Open a Numpy thing
        else
        return data

    def write_out(self, output_path):
        with open(self.input_path) as f:
            pass

    def generate_prices_at_time(self, seconds = 60):
        pass

if __name__ == "__main__":
    dataset = "D:\Data\Crypto\GDAX\BTC-USD.csv"
    myData = TradeData(dataset, rows = 1000)
    print(myData.data)