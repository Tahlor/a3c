from exchange import *
import numpy as np

DATA = ".\data\BTC_USD_100_FREQ.npy"
myExchange = Exchange(DATA, game_length=100)
print(myExchange.price_changes.shape) #34 MILLION rows total - skip first million?


for i in range(20001, 20110):
    print(myExchange.vanilla_prices[i])