#!/usr/bin/env python3

"""
File: metacurvature_fc100.py
Author: Seb Arnold - seba1511.net
Email: smr.arnold@gmail.com
Github: seba-1511
Description:
Demonstrates how to use the GBML wrapper to implement MetaCurvature.

A demonstration of the low-level API is available in:
    examples/vision/anilkfo_cifarfs.py
"""

import random
import numpy as np
import torch
import learn2learn as l2l
from learn2learn.optim.transforms import MetaCurvatureTransform

from model import CifarCNN
import sys
from tqdm import tqdm


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


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
    problem='fc100',
    fast_lr=0.1,
    meta_lr=0.01,
    num_iterations=10000,
    meta_batch_size=500,
    adaptation_steps=5,
    shots=5,
    ways=5,
    cuda=1,
    seed=1234,
    statepath=None
    ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')


    if problem.lower() == 'fc100':
        print('Dataset: FC100')
        dataroot = './data/fc100'
        model = CifarCNN(ways)
        model.to(device)

    elif problem.lower() == 'cifarfs':
        print('Dataset: CIFAR-FS')
        dataroot = './data/cifarfs'
        model = CifarCNN(ways)
        model.to(device)

    elif problem.lower() == 'mini-imagenet':
        print('Dataset: miniImageNet')
        dataroot = './data/imagenet'
        model = l2l.vision.models.MiniImagenetCNN(ways)
        model.to(device)


    # Create Tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets(
        name=problem,
        train_samples=2*shots,
        train_ways=ways,
        test_samples=2*shots,
        test_ways=ways,
        root=dataroot,
    )

    gbml = l2l.algorithms.GBML(
        model,
        transform=MetaCurvatureTransform,
        lr=fast_lr,
        adapt_transform=False,
    )
    gbml.to(device)
    opt = torch.optim.Adam(gbml.parameters(), meta_lr)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    state = torch.load(statepath)
    gbml.load_state_dict(state)

    meta_test_error = []
    meta_test_accuracy = []
    for task in tqdm(range(meta_batch_size)):
        # Compute meta-testing loss
        learner = gbml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error.append(evaluation_error.item())
        meta_test_accuracy.append(evaluation_accuracy.item())

    print('Meta Test Error', np.mean(meta_test_error), np.std(meta_test_error))
    print('Meta Test Accuracy', np.mean(meta_test_accuracy), np.std(meta_test_accuracy))



if __name__ == '__main__':

    problem = str(sys.argv[1])
    shots = int(sys.argv[2])
    statepath = str(sys.argv[3])

    main(problem=problem, shots=shots, statepath=statepath)