import numpy as np
import tensorflow as tf
import math

class Exchange:
    def __init__(self, data_stream, cash = 10000, holdings = 0, actions = [-1,1]):

        # Expects a list of dictionaries with the key price
        self.data = np.load(data_stream)
        self.state = 1
        self.cash = cash
        self.holdings = holdings
        self.actions = actions

    def get_next_state(self):
        self.state += 1
        self.current_price = self.data[self.state]["price"]
        return self.data[self.state]

    def buy_security(self, coin = None, currency = None):
        assert (coin is None) != (currency is None)

        if currency is None:
            cost = min(self.cash, self.price * coin)
        else:
            cost = min(self.cash, currency)

        self.cash -= cost
        self.holdings -= cost/self.price

    def sell_security(self, coin = None, currency = None):
        assert (coin is None) != (currency is None)

        if coin is None:
            proceeds = min(self.holdings, currency/self.price)
        else:
            proceeds = min(self.holdings, coin)

        self.cash += proceeds
        self.holdings -= proceeds/self.price

    def get_balances(self):
        return {"cash":self.cash, "holdings":self.holdings}

    def get_value(self):
        return self.cash + self.holdings*self.price

    # maybe feed absolute price and price % change from previous state
    def get_perc_change(self):
        return self.current_price/self.data[self.state-1]["price"]
        
    def interpret_action(self, action):
        # action can be a vector -1 = 1
        action = 2*(action-np.average(self.actions))/(max(self.actions)-min(self.actions))
        if action < 0:
            self.sell_security(coin = self.holdings * abs(action))
        elif action > 0:
            self.buy_security(currency = self.cash * abs(action))