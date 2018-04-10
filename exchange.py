import numpy as np
import tensorflow as tf
import math
from process_data.utils import *

DATA = ".\data\BTC_USD_100_FREQ.npy"
#DATA = ".\data\BTC-USD_VERY_SHORT.npy"

# Observe every state, but only act every few states?
# Delay moves by several states

# other features: weight losses more (to simulate risk aversion)
# use a 2 second delay before transactions to simulate latency
# Does the system need to learn it can't buy more coins if it has none? E.g. we could veto these trades;
# or we could assume it's just saying "if I could buy, I would"; in either case, it will need to learn that
# choosing a buy action without cash doesn't do anything, I think it should

# State
# Use LSTM to remember previous states
# New information is: holdings, cash, price, % change in price, whether it was a market buy/sell
# If using transaction level, can also include size of order

# Transaction costs
# Impose a transaction cost for market orders?
# OR REQUIRE the model to take limit orders
# ACTIONS ARE LIMIT ORDERS

# Instead of policy/value, we know the best move at every instant -- tell it to do that instead


# time_interval - each state is a X second period

class Exchange:
    def __init__(self, data_stream, cash = 10000, holdings = 0, actions = [-1,1], time_interval = None, transaction_cost = 0):

        '''
        Expects a list of dictionaries with the key price
            Can control network behavior with two main parameters:
                1. how long to look back (e.g., past hour, past day, etc.)
                2. how often to sample prices (e.g., get price every minute, get price every hour, etc.)
                3. maybe get order book?
        '''
        # Game parameters
        self.number_of_input_prices_for_basic = 10
        self.number_of_inputs_for_basic = self.number_of_input_prices_for_basic # *2 # for prices and positions
        self.basic_sample_frequency = 100

        self.game_length = 1000

        self.data = np.load(data_stream)
        self.state = 1
        self.starting_cash = cash
        self.cash = cash
        self.holdings = holdings
        self.actions = actions
        self.transaction_cost = transaction_cost
        self.generate_log_prices()
        self.permit_short = False
        if not time_interval is None:
            print(self.data[0:30])
            self.generate_prices_at_time(time_interval)
            self.data = self.prices_at_time

    def get_model_input(self, batch_size=1, price_range=None, exogenous=True):
        if price_range is None:
            price_range = [self.state]

        # This can be batched
        if exogenous:
            to_return = []
            for _ in range(batch_size):
                prices = np.log(self.data[slice(*price_range)]["price"])
                position = self.data[slice(*price_range)]["side"]

                to_return.append(prices)

            ### FINISH
            return np.asarray(to_return)
        else:
            return self.price_change, self.holdings, self.cash, self.data[self.state]["side"]

    def generate_log_prices(self, distance = 1):
        # distance - comparison price; e.g. 5 implies compare this price to the price 5 transactions ago
        # create log prices
        # current price - previous price
        self.price_changes = np.log(self.data[:]["price"]) * 1000.
        self.price_changes = self.price_changes[distance:] - self.price_changes[:-distance]
        self.price_changes = np.insert(self.price_changes, 0, 0) # no change for first state
        return self.price_changes

    def get_next_state(self):
        self.state += 1
        self.current_price = self.data[self.state]["price"]
        self.price_change = self.price_changes[self.state]
        return self.data[self.state]

    def goto_state(self, state):
        self.state = state
        self.current_price = self.data[self.state]["price"]
        return self.data[self.state]

    def is_terminal_state(self):
        return self.state >= len(self.data)

    # get history - n = how many rows to get, freq = how often to get them
    def get_price_history_no_batch(self, current_id = None, n = 100, freq=100):
        if current_id is None:
            current_id = n*freq
        elif current_id < n * freq:
            print("Initial trade id must be greater than freq * n")
        return np.copy(self.data[current_id:current_id-(n*freq):-freq]["price"])

    def get_price_history(self, current_id=None, n=100, freq=100, batch= True, ):
        if current_id is None:
            current_id = self.state
        if not batch:
            return self.get_price_history_no_batch(self, current_id=current_id, n=n, freq=freq)
        else:
            # For basic model, return a tensor of previous inputs
            # Array of data

            # Override default freq
            freq = self.number_of_input_prices_for_basic

            list_of_prices = self.generate_log_prices(distance = freq) # price change since last input -- freq can be anything though
            # Pattern (every 10th time)
            m = np.array(range(current_id, current_id + self.game_length, freq))
            np.tile(list_of_prices[m], (self.game_length, 1)) + np.tile(list_of_prices[:, None], freq) - 2 * self.game_length


    # same as above, but can optionally define a list [0,10,50,100] of previous time steps, or a function
    def get_price_history_func(self, current_id = None, n = 100, pattern=lambda x: x**2):
        if type(pattern) == type([]):
            if np.sum(pattern) > 0:
                pattern = -pattern
        else:
            func = pattern
            pattern = []
            for x in range(0,n):
                pattern.append(current_id-func(x))
        return np.copy(self.data[pattern]["price"])

    # look at prices every X seconds (rather than each transaction as a new state)
    def generate_prices_at_time(self, seconds = 60, prices_only = False, interpolation = "repeat"):
        current_time = self.data[0]["time"]
        target = round_to_nearest(current_time, round_by=seconds)
        previous_target = target
        self.prices_at_time = [0]

        for n, i in enumerate(self.data):
            if i["time"] > target:
                target = round_to_nearest(i["time"], seconds)
                time_steps = int((target-previous_target)/seconds ) # number of missing time intervals

                # Return list of prices only or index of complete transactions
                next_item = [n] if not prices_only else [i["price"]]

                # Interpolation if no transactions in interval
                if interpolation == "repeat":
                    self.prices_at_time += [self.prices_at_time[-1]]*time_steps + next_item
                elif interpolation is None:
                    self.prices_at_time += [None] * time_steps + next_item

                previous_target = target
                target += seconds

        self.prices_at_time.pop(0)

        if not prices_only:
            #print(self.prices_at_time[0:30])
            self.prices_at_time = np.copy(self.data[self.prices_at_time])

    def buy_security(self, coin = None, currency = None):
        assert (coin is None) != (currency is None)

        if currency is None:
            cost = min(self.cash, self.current_price * coin)
        else:
            cost = min(self.cash, currency)

        self.cash -= cost
        self.holdings += (cost * (1-self.transaction_cost)) / self.current_price

    def sell_security(self, coin = None, currency = None):
        assert (coin is None) != (currency is None)

        if coin is None:
            proceeds = min(self.holdings*self.current_price, currency) if not self.permit_short else currency
        else:
            proceeds = min(self.holdings*self.current_price, coin*self.current_price) if not self.permit_short else coin*self.current_price

        self.cash += proceeds * (1-self.transaction_cost)
        self.holdings -= proceeds/self.current_price

    def get_balances(self):
        return {"cash":self.cash, "holdings":self.holdings}

    def get_value(self):
        return self.cash + self.holdings*self.current_price

    # maybe feed absolute price and price % change from previous state
    def get_perc_change(self):
        return self.current_price/self.data[self.state-1]["price"]
        
    # needs to be a value between -1 and 1
    def interpret_action(self, action, sd, continuous = True):
        # this normalizes action to [min, max]
        if continuous:
            action = 2*(action-np.average(self.actions))/(max(self.actions)-min(self.actions))
            action = self.sample_from_action(action, sd)

        # Margin call
        if self.permit_short and self.get_value() < .1*self.starting_cash and self.holdings < 0:
            # close all negative positions if value < 1000
            self.buy_security(coin=-self.holdings)
            return action

        if action < 0:
            if not self.permit_short:
                self.sell_security(coin = self.holdings * abs(action))
            else: # if agent can short, he can short all but 20% of his initial balance
                self.sell_security(coin=(self.get_value()   -.2*self.starting_cash)/self.current_price * abs(action))
        elif action > 0:
            self.buy_security(currency = self.cash * abs(action))
        return action

    def sample_from_action(self, mean = 0, sd = 1):
        sample = np.random.normal(mean, sd)
        return min(max(sample, -1), 1)


def test_buying_and_selling(myExchange):
    #print(x)
    # action can be a vector -1 = 1
    action = 2*(action-np.average(self.actions))/(max(self.actions)-min(self.actions))
    if action < 0:
        self.sell_security(coin = self.holdings * abs(action))
    elif action > 0:
        self.buy_security(currency = self.cash * abs(action))

def test_getting_prices(myExchange):
    #x = myExchange.get_price_history_func(10000)

    x = myExchange.generate_log_prices(10)
    print(x)

if __name__ == "__main__":
    # myExchange = Exchange(DATA, time_interval=60)
    myExchange = Exchange(DATA)
    myExchange.state = 10000
    print(myExchange.get_price_history(n = 1, freq=1))
    test_getting_prices(myExchange)


