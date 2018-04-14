from exchange import *
import numpy as np

DATA = ".\data\BTC_USD_100_FREQ.npy"
myExchange = Exchange(DATA, game_length=100)
print(myExchange.price_changes.shape) #34 MILLION rows total - skip first million?

print(myExchange.vanilla_prices[20001])