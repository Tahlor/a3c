from exchange import *
import numpy as np

DATA = ".\data\BTC_USD_100_FREQ.npy"
myExchange = Exchange(DATA)
print(myExchange.price_changes.shape) #34 MILLION rows total - skip first million?