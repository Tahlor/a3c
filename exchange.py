import numpy as np
import tensorflow as tf
import math
from process_data.utils import *

DATA = ".\data\BTC_USD_100_FREQ.npy"

class Exchange:
    def __init__(self, data_stream, game_length, cash = 10000, holdings = 0, actions = [-1,1], time_interval = None,
                 transaction_cost = 0, permit_short = False, naive_inputs = 3, naive_price_history =3, naive = True):
        self.naive = naive
        self.step_counter = 1
        self.data_stream = data_stream
        # Game parameters
        self.number_of_input_types = naive_inputs
        self.number_of_input_prices_for_basic = naive_price_history
        self.total_number_of_inputs_for_basic = self.number_of_input_prices_for_basic * naive_inputs # for prices and positions
        self.game_length = game_length
        self.naive_sample_pattern = [2**x for x in range(2,2+self.number_of_input_prices_for_basic)] # for naive model, which previous prices to look at;
        # will be the len of the number of input prices, we'll add the 0 later
        self.data = np.load(data_stream)
        self.vanilla_prices = self.data[:]["price"].astype('float64')
        print("States in data: {}".format(len(self.vanilla_prices)))
        self.log_prices = np.log(np.copy(self.data[:]["price"].astype("float64")))
        self.log_price_changes = self.log_prices[1:] - self.log_prices[0:-1]
        self.gru_prime_length = self.game_length

        min_state = min(self.naive_sample_pattern + [self.gru_prime_length, 10])  # don't start at 0, make sure we can go back in time etc.
        max_state = len(self.log_prices) - self.game_length - 10  # give us a small buffer
        self.state_range = [min_state, max_state]
        self.state = min_state
        self.current_price = self.vanilla_prices[self.state]
        self.naive_prices = self.get_model_input_naive()


        self.starting_cash = cash
        self.cash = cash
        self.holdings = holdings
        self.actions = actions
        self.transaction_cost = transaction_cost
        self.price_changes = self.generate_log_prices(1, [0,len(self.data)]) # these are the price changes for the entire exchange
        self.price_change = self.price_changes[0]
        self.permit_short = permit_short
        self.margin_call = 0
        self.margin_requirement = 0

    def step(self, a):
            previous_value = self.get_value()
            self.interpret_action(a, sample=False)
            self.get_next_state() # go to next state to find reward of that move
            current_value = self.get_value()

            if False:
                # This way doesn't give rewards often enough
                R = max(current_value-self.max_value, 0)
                if R > 0:
                    self.max_value=current_value
            else:
                R = (np.log(current_value) - np.log(previous_value))*10000
                R = current_value - previous_value
                # This one gives arbitrary award amounts for not screwing up
                if False:
                    if R > 0:
                        R = max(R+50, 2*R)

            self.step_counter += 1
            return self.get_complete_state(), R, self.step_counter, 0

    def copy(self):
        new_exchange = Exchange(data_stream=self.data_stream, game_length=self.game_length)
        new_exchange.__dict__ = self.__dict__.copy()
        return new_exchange

    def reset(self, state = None):
        if state is None:
            state = self.state_range[0]
        self.step_counter = 0
        self.cash = self.starting_cash
        self.max_value = self.cash
        self.holdings = 0
        self.goto_state(state)
        self.naive_prices = self.get_model_input_naive()
        return self.get_complete_state()

    def get_complete_state(self):
        if self.naive and self.step_counter < self.game_length:
            price_change = self.naive_prices[0, self.step_counter]
        else:
            price_change = np.asarray(self.log_prices[self.state] - self.log_prices[self.state-1])
        ratio = self.get_perc_cash()
        time_increment = self.data[self.state]["time"]-self.data[self.state-1]["time"]
        side = self.data[self.state]["side"]
        complete_state = np.concatenate([price_change.reshape(-1), np.asarray([ratio, time_increment, side])])
        return complete_state

    def get_model_input(self, batch_size=1, price_range=None, exogenous=True):
        if price_range is None:
            price_range = [self.state]

        # This can be batched
        if exogenous:
            to_return = []
            for _ in range(batch_size):
                #prices = np.log(self.data[slice(*price_range)]["price"])
                prices = self.generate_log_prices()
                positions = self.data[slice(*price_range)]["side"]

                combined = np.empty([prices.size + positions.size], dtype=prices.dtype)
                combined[0::2] = prices
                combined[1::2] = positions

                to_return.append(combined)

            return np.asarray(to_return)
        else:
            return self.price_change, self.holdings, self.cash, self.data[self.state]["side"]

    def generate_log_prices(self, distance=1, state_range=None):
        # distance - comparison price; e.g. 5 implies compare this price to the price 5 transactions ago
        # 1 is the previous price
        # create log prices
        # current price - previous price
        if state_range is None:
            # generate price changes for game
            if self.state > 0:
                # in this case we have to get price one step before the game starts
                # so we can have a valid price change for the first state in the game
                # otherwise this list ends up being GAME_LENGTH - 1 long and things break
                state_range = [self.state-distance, self.state + self.game_length]
            else:
                # if game starts at beginning of data, we'll insert a 0 later to make the size of the list work out
                state_range = [self.state, self.state + self.game_length]  # generate price changes for game
        backsteps = min(distance, state_range[0]) # can't go before beginning of time
        state_range = [x - backsteps for x in state_range]
        price_changes = np.log(self.data[slice(*state_range)]["price"].astype('float64')*1.0) #np.log
        # print(type(price_changes))
        price_changes = price_changes[distance:] - price_changes[:-distance]


        price_changes = np.insert(price_changes, 0, [0] * (distance-backsteps)) # no change for first state;

        return price_changes

    def get_next_state(self):
        self.state += 1
        self.current_price = self.vanilla_prices[self.state] #self.data[self.state]["price"]
        self.price_change = self.price_changes[self.state] # use the log price changes

        # Round off
        self.cash = round(self.cash, 5)
        #print("CASH {}".format(self.cash))
        self.holdings = round(self.holdings, 5)
        return self.data[self.state]

    def goto_state(self, state):
        self.state = state
        self.current_price = self.data[self.state]["price"]
        return self.data[self.state]

    def is_terminal_state(self):
        return self.state >= len(self.data)

    # get history - n = how many rows to get, freq = how often to get them
    # this is only good for a single step in one game in the basic model
    def get_price_history_step(self, current_id = None, n = 100, freq=100):
        if current_id is None:
            current_id = n*freq
        elif current_id < n * freq:
            print("Initial trade id must be greater than freq * n")
        return np.copy(self.data[current_id:current_id-(n*freq):-freq]["price"])

    def get_batch_price_indices(self, state_range, freq, backsamples = None):
        if state_range is None:
            state_range = [self.state, self.state + self.game_length]
        if backsamples is None:
            backsamples = self.number_of_input_prices_for_basic

        # Handle a list input - list should be of form [1,4,9] to bring in prices from 4th previous price, 9th previous price, etc.
        if type(freq) == type([]):
            backsamples = len(freq)
            further_back_ref = max(freq)

            # Add a 0 if needed
            if freq[0] != 0:
                pattern = np.insert(0 - np.array(freq), 0, 0)
            else:
                pattern = freq
        else:
            further_back_ref = freq * backsamples # e.g. the earliest price I need
            pattern = np.array(range(0, -further_back_ref - 1, -freq))

        # Pattern will now look like [0, -1, -4 etc.]

        ### further_back_ref needs to be greater than start! ###

        size = state_range[1]-state_range[0]  # size of game
        start = state_range[0]
        end = start + size

        m = range(start - further_back_ref, end)
        x = np.asarray(range(start - further_back_ref, end)) # create an array starting at the earliest index value you need

        # [backsamples-1::-1][::-1]
        z = np.tile(x[pattern + further_back_ref], (size, 1)) + np.tile(np.array(range(0, size))[:, None], backsamples + 1)
        return z

    # This will return a single games worth of steps for the basic model
    # There is no "batching" however
    # Return a [1 (batch size) x seq length x # of input prices)
    def get_price_history(self, prices=None, start=None, state_range = None, backsamples=None, freq=None, batch= True, calc_diff = True):
        if prices is None:
            prices = self.log_prices
        if range is None and start is None:
            state_range = [self.state, self.state+self.game_length]
        if freq is None:
            freq = self.game_length/10
        if backsamples is None:
            backsamples = self.number_of_input_prices_for_basic

        if not batch:
            return self.get_price_history_step(self, current_id=start, n=backsamples, freq=freq)
        else:
            # For basic model, return a tensor of previous inputs
            # Array of data

            # Override default freq
            # There are two frequencies -- one is the frequency of previous time steps (e.g. for the naive model)
            # Other frequency is the comparison price for % change calculation - kind of a gradient over that period
            price_indices = self.get_batch_price_indices(state_range=state_range, freq=freq, backsamples=backsamples)

            # Constant shift
            #comparison_prices_for_game = self.get_batch_prices(prices=self.log_prices, state_range=state_range-freq, freq=freq, backsamples=backsamples)
            # (prices_for_game - comparison_prices_for_game)[:, None]
            # Relative shift:
            if not calc_diff:
                return prices[price_indices]
            else:
                return prices[price_indices][:,0:-1] - prices[price_indices][:,1:]

    def whiten(self, x):
        return (x - np.mean(x)) / math.sqrt(np.var(x))

    def get_model_input_naive(self, whiten=True):
        # this uses current state
        prices = self.get_price_history(freq=self.naive_sample_pattern, batch= True, )[None,...]
        if self.number_of_input_types == 2:
            buy_sell_indices = self.get_batch_price_indices(state_range = None, freq=self.naive_sample_pattern, ) [:,:-1] # n-1 in prices since using the difference
            buy_sell = self.data[:]["side"][buy_sell_indices] [None,...] # add a batch dimension
            network_input = np.concatenate((prices,buy_sell), 2) #[1 (batches x seq length x prev_states * 2)] ; 2 is for prices and sides
        else:
            network_input= prices
        #basic = np.asarray(self.vanilla_prices[self.state:self.state+self.game_length]).reshape([1,-1,1])
        #basic = (basic - np.mean(basic)) #/(np.max(basic)-np.min(basic)) # normalize
        if whiten:
            network_input = self.whiten(network_input)
        return network_input

    def buy_security(self, coin = None, currency = None):
        assert (coin is None) != (currency is None)

        if currency is None:
            cost = min(self.cash, self.current_price * coin)
        else:
            cost = min(self.cash, currency)

        # buy an even amount of coins
        coins_bought = round((cost * (1-self.transaction_cost)) / self.current_price, 2)
        self.cash -= coins_bought * self.current_price * (1+self.transaction_cost)
        self.holdings += coins_bought

    def sell_security(self, coin = None, currency = None):
        assert (coin is None) != (currency is None)

        if coin is None:
            proceeds = min(self.holdings*self.current_price, currency) if not self.permit_short else currency
        else:
            proceeds = min(self.holdings*self.current_price, coin*self.current_price) if not self.permit_short else coin*self.current_price

        coins_sold = round(proceeds/self.current_price, 2)
        self.cash += coins_sold * self.current_price * (1-self.transaction_cost)
        self.holdings -= coins_sold

    def get_balances(self):
        return {"cash":self.cash, "holdings":self.holdings}

    def get_value(self):
        return self.cash + self.holdings*self.current_price

    def get_profit(self):
        return self.get_value() - self.starting_cash

    def get_perc_cash(self):
        return self.cash/self.get_value()

    # maybe feed absolute price and price % change from previous state
    def get_perc_change(self):
        return self.current_price/self.data[self.state-1]["price"]

    def interpret_action(self, action, sd=0, continuous = True, sample = True):
        # this normalizes action to [min, max]
        raw_action = action[0]
        action = round( min(max(raw_action, -1), 1), 2) # round off, put action in acceptable range

        # Margin call
        if self.permit_short and self.get_value() < self.margin_call*self.starting_cash and self.holdings < 0:
            # close all negative positions if value < 1000
            self.buy_security(coin=-self.holdings)
            return action

        if action < 0:
            if not self.permit_short:
                self.sell_security(coin = self.holdings * abs(action))
            else: # if agent can short, he can short all but 20% of his initial balance
                self.sell_security(coin=((self.get_value() - self.margin_requirement*self.starting_cash)/self.current_price) * abs(action))
        elif action > 0:
            self.buy_security(currency = self.cash * abs(action))
        #print(action)
        #print(self.get_status(), action)
        return raw_action # don't tell the model we rounded the recommendation

    def get_status(self):
        print("Cash {}, Holdings {}, Price {}, State {}".format(self.cash, self.holdings, self.current_price, self.state))

if __name__ == "__main__":
    #np.set_printoptions(formatter={'int_kind': lambda x: "{:0>3d}".format(x)})
    #np.set_printoptions(formatter={'float_kind': lambda x: "{0:6.3f}".format(x)})

    # myExchange = Exchange(DATA, time_interval=60)
    # myExchange = Exchange(DATA)
    # myExchange.state = 10000
    # test_getting_prices(myExchange)
    #print(myExchange.get_price_history(n = 1, freq=1))

    #play_game()
    pass


