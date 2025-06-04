#!/usr/bin/env python3

import random
import numpy as np
import torch
import learn2learn as l2l
from learn2learn.optim.transforms import MetaCurvatureTransform

from model import CifarCNN
import sys
from tqdm import tqdm
from copy import deepcopy


class AccDS:
    def __init__(self, acctype, indices):

        self.acctype = acctype
        self.acc = 0.0 if acctype == 'max' else float("inf")
        self.indices = indices
    
    def add(self, acc, indices):

        if (self.acctype == 'min' and acc < self.acc) or (self.acctype == 'max' and acc > self.acc):
            self.acc = acc
            self.indices = deepcopy(indices)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    acc = (predictions == targets).sum().float() / targets.size(0)
    return acc * 100.0


def fast_adapt(batch, batch_eval, learner, loss, adaptation_steps, shots, ways, device):
    adaptation_data, adaptation_labels = batch
    adaptation_data, adaptation_labels = adaptation_data.to(device), adaptation_labels.to(device)

    evaluation_data, evaluation_labels = batch_eval
    evaluation_data, evaluation_labels = evaluation_data.to(device), evaluation_labels.to(device)

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
    statepath=None,
    acctype='min',
    attack_passes=3
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

    dataset = tasksets.test.dataset

    meta_test_error = []
    meta_test_accuracy = []
    acc_ds_list = []

    n_test, n_train = 200, 400


    for seed1 in tqdm(range(1), desc="Seed", leave=False):
        random.seed(seed + seed1)
        np.random.seed(seed + seed1)
        torch.manual_seed(seed + seed1)

        # sample classes
        labels = random.sample(dataset.labels, ways)

        test_idxs, test_labels = [], []
        train_idxs, train_labels = [], []

        for labelidx, label in enumerate(labels):
            idxs = np.random.permutation(dataset.labels_to_indices[label])
            test_idxs.extend(idxs[:n_test])
            test_labels.extend([labelidx] * n_test)

            train_idxs.append(idxs[n_test:])
            train_labels.append([labelidx] * n_train)

        # Create test set
        test_X = torch.stack([dataset[idx][0] for idx in test_idxs], dim=0)
        test_Y = torch.LongTensor(test_labels)

        perm = np.random.permutation(test_X.shape[0])
        test_X = test_X[perm]
        test_Y = test_Y[perm]

        batch_eval = (test_X, test_Y)
        
        imgidxs = {label: random.sample(list(idxs), shots) for label, idxs in enumerate(train_idxs)}
        acc_ds = AccDS(acctype, imgidxs)

        passbar = tqdm(range(attack_passes), desc="Attack pass", leave=False)
        for passidx in passbar:
            for clsidx in tqdm(range(ways), desc="nWay", leave=False):
                for shotidx in tqdm(range(shots), desc="nShots", leave=False):
                    imgidxs = deepcopy(acc_ds.indices)
                    available_idxs = [x for x in train_idxs[clsidx] if x not in imgidxs[clsidx]] + [imgidxs[clsidx][shotidx]]

                    for idx in tqdm(available_idxs, desc="Img idx", leave=False):
                        imgidxs[clsidx][shotidx] = idx

                        # Create dataset
                        adapt_idxs, adapt_labels = [], []
                        for label, idxs in imgidxs.items():
                            adapt_idxs.extend(idxs)
                            adapt_labels.extend([label] * len(idxs))

                        train_X = torch.stack([dataset[idx][0] for idx in adapt_idxs], dim=0)
                        train_Y = torch.LongTensor(adapt_labels)

                        perm = np.random.permutation(train_X.shape[0])
                        train_X = train_X[perm]
                        train_Y = train_Y[perm]

                        batch_adapt = (train_X, train_Y)

                        learner = gbml.clone()
                        eval_error, eval_acc = fast_adapt(
                            batch_adapt, batch_eval, learner, loss, adaptation_steps, shots, ways, device
                        )

                        acc_ds.add(eval_acc.item(), imgidxs)
                        passbar.set_postfix(min_acc=acc_ds.acc, last_acc=eval_acc.item())

        acc_ds_list.append(acc_ds)
        accs = [acc_ds.acc for acc_ds in acc_ds_list]
        print(f'Mean Adv Test accuracy: {np.mean(accs)}. Std: {np.std(accs)}')
        sys.stdout.flush()

    accs = [acc_ds.acc for acc_ds in acc_ds_list]
    print(f'Mean Adv Test accuracy: {np.mean(accs)}. Std: {np.std(accs)}')


if __name__ == '__main__':

    problem = str(sys.argv[1])
    shots = int(sys.argv[2])
    statepath = str(sys.argv[3])
    seed = int(sys.argv[4])
    acctype = str(sys.argv[5])

    main(problem=problem, shots=shots, statepath=statepath, seed=seed, acctype=acctype)