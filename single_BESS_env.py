import random
import numpy as np
import pandas as pd
import gym
from gym import spaces
import math

# Default parameters for default MG EMS environment.
# days range
DEFAULT_DAY0 = 0
DEFAULT_DAYN = 10
# Balancing market prices
DEFAULT_MK_PRICE = list(pd.read_csv('AESO_price.csv', index_col=None, skiprows=4).to_numpy()[:,1])
# get the past 24 hours price of the first hour of the first day
temp_price = []
for _ in range(24):
    temp = DEFAULT_MK_PRICE.pop()
    temp_price.append(temp)
DEFAULT_PAST_MK_PRICE = temp_price[::-1]
# Length of one episode
DEFAULT_ITERATIONS = 24
# Battery characteristics (MWh)
DEFAULT_BAT_CAPACITY = 5
DEFAULT_MAX_CHARGE = 3  # (MW)
DEFAULT_MAX_DISCHARGE = 3  # (MW)
DEFAULT_BAT_EFFICIENCY = 0.95
DEFAULT_BAT_DEC = 1.2  # degradation cost of battery charging and discharging $/kWh
DEFAULT_MAX_SOC = 0.9
DEFAULT_MIN_SOC = 0.1
DEFAULT_INI_CAP = 0.4
# normalizer for reward
DEFAULT_MAX_REWARD = 1000
# Rendering lists
BATTERY_RENDER = []  # battery within the MG
PRICE_RENDER = []  # market price


class Battery:  # Simulates the battery system of the microGrid
    def __init__(self, max_cap, max_charge, max_discharge, efficiency, max_soc, min_soc, ini_cap, de_cost):
        self.max_cap = max_cap  # fully charged battery capacity
        self.ini_cap = ini_cap  # initial capacity
        self.max_charge = max_charge  # max charge power
        self.max_discharge = max_discharge  # max discharge power
        self.efficiency = efficiency  # charging and discharging efficiency
        self.max_soc = max_soc  # max SOC
        self.min_soc = min_soc  # min SOC
        self.de_cost = de_cost  # degradation cost of battery

    def charge_cost(self, energy):  # return charging degradation cost
        temp_energy = energy
        if energy > self.max_charge:
            temp_energy = self.max_charge
        if energy < 0:
            temp_energy = 0
        self.current_cap = (self.current_cap * self.max_cap + temp_energy * self.efficiency) / self.max_cap
        return temp_energy ** 2 * self.de_cost

    def discharge_cost(self, energy):  # return discharging degradation cost
        temp_energy = energy
        if energy > self.max_discharge:
            temp_energy = self.max_discharge
        if energy < 0:
            temp_energy = 0
        self.current_cap = (self.current_cap * self.max_cap - temp_energy / self.efficiency) / self.max_cap
        return temp_energy ** 2 * self.de_cost

    def dissipate(self):
        self.current_cap *= math.exp(- self.dissipation)

    def SOC(self):
        return self.current_cap

    def reset(self):
        self.current_cap = self.ini_cap


class Grid:  # market prices
    def __init__(self, price: list, past_price: list):
        self.price = price
        self.past_price = past_price  # 24 hours price before day 0

    def sell(self, energy):  # revenue from selling energy
        return self.current_price * energy

    def buy(self, energy):  # cost for purchasing energy
        return self.current_price * energy

    def set_time(self, day, time):  # set current time for the grid env
        self.time = time
        self.day = day
        self.current_price = self.price[self.day*24+self.time]

    def retrieve_past_price(self):  # return the price of last 24 hours
        result = []
        if self.day < 1:  # generate last day prices
            past_price = self.past_price
        else:
            past_price = self.price[24*(self.day-1):24*self.day]
        for item in past_price[(self.time-24)::]:
            result.append(item)
        for item in self.price[24*self.day:(24*self.day+self.time)]:
            result.append(item)
        return result


class BESSEnv(gym.Env):
    def __init__(self, **kwargs):
        super(BESSEnv, self).__init__()
        # parameters (we have to define it through kwargs because of how Gym works...)
        self.iterations = kwargs.get("iterations", DEFAULT_ITERATIONS)
        self.day0 = kwargs.get("day0", DEFAULT_DAY0)
        self.dayn = kwargs.get("dayn", self.day0 + 1)
        self.mk_price = kwargs.get("mk_price", DEFAULT_MK_PRICE)
        self.past_mk_price = kwargs.get("past_mk_price", DEFAULT_PAST_MK_PRICE)
        self.bat_capacity = kwargs.get("battery_capacity", DEFAULT_BAT_CAPACITY)
        self.bat_max_charge = kwargs.get("max_charge", DEFAULT_MAX_CHARGE)
        self.bat_max_discharge = kwargs.get("max_discharge", DEFAULT_MAX_DISCHARGE)
        self.bat_efficiency = kwargs.get("efficiency", DEFAULT_BAT_EFFICIENCY)
        self.bat_max_soc = kwargs.get("max_soc", DEFAULT_MAX_SOC)
        self.bat_min_soc = kwargs.get("min_soc", DEFAULT_MIN_SOC)
        self.bat_ini_cap = kwargs.get("ini_cap", DEFAULT_INI_CAP)
        self.bat_de_cost = kwargs.get("de_cost", DEFAULT_BAT_DEC)
        self.day = self.day0  # count episode
        self.current_time = 0  # current hour index, start from 0
        self.max_reward = kwargs.get("max_reward", DEFAULT_MAX_REWARD)
        # individual components of the environment
        self.grid = Grid(self.mk_price, self.past_mk_price)
        self.battery = Battery(max_cap=self.bat_capacity, max_charge=self.bat_max_charge,
                               max_discharge=self.bat_max_discharge,
                               efficiency=self.bat_efficiency, max_soc=self.bat_max_soc, min_soc=self.bat_min_soc,
                               ini_cap=self.bat_ini_cap, de_cost=self.bat_de_cost)
        # define normalized action space
        # (battery_charge, battery_discharge) :binary
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        # define normalized state space (observations)
        # market price of past 24 hours, soc of bat
        self.state_space = spaces.Box(low=0, high=1, shape=(24 + 1,), dtype=np.float32)

    def reset(self, day=None):  # returns an initial observation
        """
        Create new TCLs, and return initial state.
        Note: Overrides previous TCLs
        create new states and override previous state
        """
        if day == None:
            self.day = self.day0
        else:
            self.day = day
        self.current_time = 0
        self.battery.reset()
        return self._build_state()

    def _build_state(self):
        """
        Return current state representation as one vector.
        Returns:
            state: 1D state vector, containing
            [current battery soc, current net power generation, current market price, current time (hour)]
            scale down to [0,1]
        """
        bat_soc = self.battery.current_cap
        # deterministic net generation and market prices (100% forecast accuracy)
        self.grid.set_time(self.day, self.current_time)
        temp_price = self.grid.retrieve_past_price()
        norm = max(temp_price)
        price_normalized = [price/norm for price in temp_price]
        obs = np.concatenate((price_normalized, bat_soc), axis=None)
        return obs

    def get_obs(self):
        """
        retrun current observation of the environment as one vector
        :return: 1D-vector
        """
        return self._build_state()

    def _take_action(self, action):  # the action will only change the battery soc
        """
        state of the env after taking one action
        :param action:
        :return: None
        """
        if action == 0:  # idle state
            ch_energy = 0
            dis_energy = 0
            pass
        elif action == 1:  # max charging
            ch_energy = self.bat_max_charge
            dis_energy = 0
        else:  # max discharging
            ch_energy = 0
            dis_energy = self.bat_max_discharge
        delta_cap = ch_energy - dis_energy
        self.battery.current_cap += delta_cap

    def _get_reward(self, action):
        reward = 0
        if action == 0:
            ch_energy = 0
            dis_energy = 0
            pass
        elif action == 1:
            ch_energy = self.bat_max_charge
            dis_energy = 0
        else:
            ch_energy = 0
            dis_energy = self.bat_max_discharge
        reward -= self.battery.charge_cost(ch_energy) \
            - self.battery.discharge_cost(dis_energy) \
            + self.grid.sell(dis_energy) \
            - self.grid.buy(ch_energy)
        reward /= self.max_reward
        # # simultaneous charging and discharging
        # if abs(ch_energy-dis_energy) > self.tolerance:
        #     reward -= self.penalty
        return reward

    def step(self, action):
        """
        the BESS EMS agent take actions, and receive reward and observe next state from the environment
        Arguments:
            action: A list.
            [battery charge power, battery discharge power, dispatchable generator output, energy import, energy export]
        Returns:
            state: state of time step t+1 (update state)
            reward: How much reward was obtained on last action
            terminal: Boolean on if the game ended (reach maximum number of iterations)
            info: None (not used here)
        """
        current_obs = self._build_state()
        self._take_action(action)
        reward = self._get_reward(action)
        self.current_time += 1
        finish = (self.current_time == self.iterations)
        if finish:  # the final state of an episode
            self.day += 1
            next_obs = self.reset(day=self.day)
        else:  # not the final state of an episode
            next_obs = self._build_state()
        return current_obs, next_obs, reward, finish

    def render(self, current_obs, next_obs, reward, finish):
        print('hour={:2d}, state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(self.current_time, current_obs, next_obs, reward, finish))

if __name__ == '__main__':  # test of the environment
    env = BESSEnv()
    rewards = []
    current_obs = env.reset()
    temp_action = [1, 0]
    for _ in range(24):
        current_obs, next_obs, reward, finish = env.step(temp_action)
        env.render(current_obs, next_obs, reward, finish)
        current_obs = next_obs
        rewards.append(reward)
    print(f"total reward {sum(rewards)}")
