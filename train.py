import argparse
import numpy as np
import os
import pickle
import scipy.stats as ss

import torch
import torch.utils.data
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from model import BayesianRanking


def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x[0])

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train
    
def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x[0])
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

def main(args):
    pyro.set_rng_seed(1219)
    pyro.clear_param_store()

    data = pickle.load(open('data/results.pkl', 'rb'))
    data = torch.tensor(data)
    train_set, test_set = data[:30000], data[30000:]
    train_set = torch.utils.data.TensorDataset(train_set)
    test_set = torch.utils.data.TensorDataset(test_set)
    kwargs = {'num_workers': 1, 'pin_memory': args.use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=args.batchsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=args.batchsize, shuffle=False, **kwargs)

    br = BayesianRanking(2000)
    optimizer = Adam({"lr": args.lr})
    svi = SVI(br.model, br.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    for epoch in range(args.epochs):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=args.use_cuda)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_freq == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=args.use_cuda)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

    mu = pickle.load(open('./data/mu_gt.pkl', 'rb'))
    pred_mu_q = pyro.param('mu_q').detach().numpy().squeeze()
    print(ss.spearmanr(mu, pred_mu_q))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', '-e', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', '-l', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--use_cuda', action='store_true')
                        
    main(parser.parse_args())