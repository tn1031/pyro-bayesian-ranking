import numpy as np
import scipy.stats as ss
from scipy.special import expit as sigmoid
import pickle

import trueskill
from trueskill import Rating, quality_1vs1, rate_1vs1


class GameManager:
    def __init__(self, n_players):
        self.n_players = n_players
        self.players = [Rating() for _ in range(n_players)]
        self.played_cnt = np.zeros(n_players)
        trueskill.backends.choose_backend('scipy')

    def _compute_draw_prob(self, i):
        mu_var = np.array([[p.mu, p.sigma**2] for p in self.players])
        beta = trueskill.BETA ** 2  # using default env
        z = 1. / (2*beta + mu_var[i,1] + mu_var[:,1])
        diff_mu = (mu_var[i,0] - mu_var[:,0]) ** 2
        _prob = np.sqrt(z) * np.exp(-0.5 * diff_mu * z)
        return _prob

    def update_and_matchmaking(self, winner_id, loser_id, is_draw,
                               is_deterministic=True):
        self.update_rating(winner_id, loser_id, is_draw)
        return self.get_next_pair(is_deterministic)

    def get_next_pair(self, is_deterministic=True):
        least_played_id = np.argmin(self.played_cnt)
        player_i = self.players[least_played_id]

        prob = self._compute_draw_prob(least_played_id)
        prob /= np.sum(prob)

        if is_deterministic:
            # because the largest is oneself.
            competitor_id = np.argsort(prob)[-2]
        else:
            # get a competitor probabilistically.
            competitor_id = np.random.choice(self.n_players, p=prob)
            while competitor_id == least_played_id:
                competitor_id = np.random.choice(self.n_players, p=prob)

        self.played_cnt[least_played_id] += 1
        self.played_cnt[competitor_id] += 1
        return least_played_id, competitor_id

    def update_rating(self, winner_id, loser_id, is_draw):
        winner = self.players[winner_id]
        loser = self.players[loser_id]
        new_rating_winner, new_rating_loser = rate_1vs1(winner, loser, is_draw)
        self.players[winner_id] = new_rating_winner
        self.players[loser_id] = new_rating_loser

def main():
    N = 2000
    mu = np.random.normal(size=N)
    sigma = np.random.gamma(1.0, 1.0, size=N)

    def compare(i, j):
        if np.random.choice([0, 1], p=[0.9, 0.1]):
            # draw
            return 0
        z_i = np.random.normal(mu[i], sigma[i])
        z_j = np.random.normal(mu[j], sigma[j])
        p = sigmoid(z_i - z_j)
        if np.random.choice([0, 1], p=[1-p, p]):
            return 1
        else:
            return 2

    game = GameManager(N)

    results = []
    for t in range(33000):
        if t == 0:
            i, j = game.get_next_pair(False)
        res = compare(i, j)
        results.append((i, j, res))
        if res == 0:
            i, j = game.update_and_matchmaking(i, j, True, False)
        elif res == 1:
            i, j = game.update_and_matchmaking(i, j, False, False)
        elif res == 2:
            i, j = game.update_and_matchmaking(j, i, False, False)
        else:
            raise

    pred_mu = np.array([p.mu for p in game.players])
    print(ss.spearmanr(mu, pred_mu))

    with open('./data/results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('./data/mu_gt.pkl', 'wb') as f:
        pickle.dump(mu, f)

if __name__ == "__main__":
    main()
    