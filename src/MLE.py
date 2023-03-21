import numpy as np
import pandas as pd
from scipy.stats import norm

class MLE(): # pretty cool surprised it actually works, but always converged on 10.828673870054516
    def __init__(self, src, max_iters=100000, lr=5):
        self.twoSigma = 11 # random initialization
        self.df = pd.read_csv(src)
        self.iters = max_iters
        self.lr = lr

        # find the difference
        difference = (self.df["team 1 score"] - self.df["team 2 score"]).to_numpy().reshape((200, 1))
        wins = self.df["winning team"].to_numpy().reshape((200, 1))
        self.df = np.hstack((difference, wins))

        # split into team1 win and team2 win
        self.win1 = self.df[self.df[:,1] == 1]
        self.win2 = self.df[self.df[:,1] == 0]
    
    def step(self):
        # during each step, find the log likelihood given self.theta, find the gradient, and update p w.r.t to gradient
        p = self.log_likelihood()
        grad = np.sum(norm(loc = self.win1[:, 0], scale = self.twoSigma).pdf(0)) - np.sum(norm(loc = self.win2[:, 0], scale = self.twoSigma).pdf(0))
        grad /= p
        return grad
    
    def log_likelihood(self):
        p1 = norm(loc = self.win1[:, 0], scale = self.twoSigma).cdf(0) # use log cdf?
        p2 = 1 - norm(loc = self.win2[:, 0], scale = self.twoSigma).cdf(0)
        return np.sum(np.log(p1)) + np.sum(np.log(p2))

    def calculate_MLE(self):
        for iter in range(self.iters):
            # print(iter, self.twoSigma)
            grad = self.step()
            self.twoSigma -= grad * self.lr

            if grad < 0.00000000001:
                break
        return self.twoSigma