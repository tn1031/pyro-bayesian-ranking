import torch
import torch.nn as nn
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist


class BayesianRanking(nn.Module):
    def __init__(self, num_players, use_cuda=False):
        self.mu_0 = 25.
        self.sigma_0 = 25. / 3
        self.beta = self.sigma_0 / 2
        self.draw_prob = .10
        self.num_players = num_players
        
        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda

    def model(self, x):
        _draw_prob = torch.tensor(self.draw_prob / (1 - self.draw_prob))

        skill_dist = dist.Normal(self.mu_0 * torch.ones(self.num_players, 1),
                                 self.sigma_0 * torch.ones(self.num_players, 1))
        skill = pyro.sample('skill', skill_dist.to_event(1))

        for t in range(len(x)):
            skill_i, skill_j = skill[x[t, 0]], skill[x[t, 1]]
            perf_diff_dist = dist.Normal(skill_i - skill_j, 2 * self.beta)
            perf_diff = pyro.sample('perf_diff_{}'.format(t), perf_diff_dist)
            pos = torch.sigmoid(perf_diff)

            result_dist = dist.Categorical(torch.cat([_draw_prob.expand_as(pos), pos, 1-pos], dim=0))
            pyro.sample('obs_{}'.format(t), result_dist, obs=x[t, 2])
        """
        with pyro.plate('matching'):
            skill_i, skill_j = skill[x[:, 0]], skill[x[:, 1]]
            perf_diff_dist = dist.Normal(skill_i - skill_j, 2 * self.beta)
            #perf_diff = pyro.sample('perf_diff_{}'.format(t), perf_diff_dist.to_event(1))
            perf_diff = pyro.sample('perf_diff', perf_diff_dist.to_event(1))
            pos = torch.sigmoid(perf_diff)

            result_dist = dist.Categorical(torch.cat([_draw_prob.expand_as(pos), pos, 1-pos], dim=1))
            #pyro.sample('obs_{}'.format(t), result_dist.to_event(1), obs=x[:, 2])
            pyro.sample('obs', result_dist.to_event(1), obs=x[:, 2][:, None])
        """

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        mu_q = pyro.param("mu_q", self.mu_0 * torch.ones(self.num_players, 1))
        sigma_q = pyro.param("sigma_q", self.sigma_0 * torch.ones(self.num_players, 1),
                             constraint=constraints.positive)
        pyro.sample('skill', dist.Normal(mu_q, sigma_q).to_event(1))
