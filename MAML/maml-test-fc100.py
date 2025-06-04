#!/usr/bin/env python3

import random
import numpy as np
import sys
from tqdm import tqdm

import torch
from torch import nn, optim
import torchvision as tv

from model import CifarCNN

import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels, FusedNWaysKShots)


class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    acc = (predictions == targets).sum().float() / targets.size(0)
    return acc * 100.0


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


def main(
        ways=5,
        shots=5,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=500,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
        statepath=None
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create datasets
    taskset = l2l.vision.benchmarks.get_tasksets('fc100',
                                                train_samples=2*shots,
                                                train_ways=ways,
                                                test_samples=2*shots,
                                                test_ways=ways,
                                                root="./data/fc100")

    # Create model
    model = CifarCNN(ways)
    model.to(device)
    
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    state = torch.load(statepath)
    maml.load_state_dict(state)

    meta_test_error = []
    meta_test_accuracy = []
    for task in tqdm(range(meta_batch_size)):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = taskset.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error.append(evaluation_error.item())
        meta_test_accuracy.append(evaluation_accuracy.item())

    acc_ci95 = 1.96 * np.std(np.array(meta_test_accuracy)) / np.sqrt(meta_batch_size)

    print('Meta Test Error. Mean: {}, Stddev: {}'.format(np.mean(meta_test_error), np.std(meta_test_error)))
    print('Meta Test Accuracy. Mean: {}, Stddev: {}, CI95: {}'.format(np.mean(meta_test_accuracy), np.std(meta_test_accuracy), acc_ci95))


if __name__ == '__main__':
    nshot = int(sys.argv[1])
    state = str(sys.argv[2])

    main(shots=nshot, statepath=state)