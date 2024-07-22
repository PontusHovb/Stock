import numpy as np
import matplotlib.pyplot as plt

import Stocks

GBG_TIME_STEP = 0.01
GBG_NUM_PATHS = 50

class GeometricBrownianMotion:
    def __init__(self, S0, mu, sigma, T, delta_t=GBG_TIME_STEP, no_paths=GBG_NUM_PATHS):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

        self.num_steps = int(T / delta_t) + 1
        self.num_paths = no_paths
        self.times = np.linspace(0, T, self.num_steps)
        self.paths = np.zeros((self.num_steps, self.num_paths))

    def __str__(self):
        self.get_paths()
        plt.figure(figsize=(9, 6))
        for i in range(self.num_paths):
            plt.plot(self.times, self.paths[:, i], color='cornflowerblue', lw=0.5)
        plt.title('Geometric Brownian Motion Paths')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()
        return f"Displayed {self.num_paths} Geometric Brownian Motion paths."

    def get_paths(self):
        self.paths[0] = self.S0

        for i in range(self.num_paths):
            dW = np.random.normal(0, np.sqrt(GBG_TIME_STEP), self.num_steps - 1)
            cumulative_dW = np.cumsum(dW)
            self.paths[1:, i] = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * self.times[1:] + self.sigma * cumulative_dW)

        return self.paths

def main():
    stock = Stocks.Stock('AAPL')
    brownian_motion = GeometricBrownianMotion(stock.price, stock.mu(365), stock.vol(365), 1)
    print(brownian_motion)

if __name__ == '__main__':
    main()