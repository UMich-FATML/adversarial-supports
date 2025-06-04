# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tqdm import tqdm

from models.protonet_embedding import ProtoNetEmbedding
from models.R2D2_embedding import R2D2Embedding
from models.ResNet12_embedding import resnet12

from models.classification_heads import ClassificationHead

from utils import pprint, set_gpu, Timer, count_accuracy, log

from data.dataloader_test import AdversarialFewShotTestDataloader

import numpy as np
import os
import random
import pickle
import copy
import sys

import seaborn as sns
import matplotlib.pyplot as plt

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network, device_ids=None)
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if opt.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()    
    elif opt.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif opt.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif opt.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print ("Cannot recognize the classification head type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase=opt.phase, do_not_use_random_transf=True)
        data_loader = AdversarialFewShotTestDataloader
    
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_test = tieredImageNet(phase=opt.phase, do_not_use_random_transf=True)
        data_loader = AdversarialFewShotTestDataloader
    
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS
        dataset_test = CIFAR_FS(phase=opt.phase, do_not_use_random_transf=True)
        data_loader = AdversarialFewShotTestDataloader

    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_test = FC100(phase=opt.phase, do_not_use_random_transf=True)
        data_loader = AdversarialFewShotTestDataloader
    
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_test, data_loader)

def destandardize(x, ds):

    if ds == 'CIFAR_FS':
        mean_pix = torch.Tensor([x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]])
        std_pix = torch.Tensor([x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]])
    elif ds == 'FC100':
        mean_pix = torch.Tensor([x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]])
        std_pix = torch.Tensor([x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]])
    elif ds == 'miniImageNet':
        mean_pix = torch.Tensor([x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]])
        std_pix = torch.Tensor([x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]])
    
    x = x.mul(std_pix).add(mean_pix)
    return x

def plot_images(all_accs, all_img_idxs, dataset, ds_name, outdir, nshot, nway):

    os.makedirs(outdir, exist_ok=True)
    all_img_idxs_sorted = sorted(all_img_idxs, key=lambda x: all_accs[x[0]])
    n_mid = int(len(all_img_idxs_sorted) / 2)

    subsampled_img_idxs_sorted = all_img_idxs_sorted[:20] + all_img_idxs_sorted[n_mid-10:n_mid+10] + all_img_idxs_sorted[-20:]

    for fidx, (acc_idxs, img_idxs) in enumerate(tqdm(subsampled_img_idxs_sorted, desc="Saving images")):
        acc = all_accs[acc_idxs]

        fig, axs = plt.subplots(nshot, nway, figsize=(nway*3, nshot*3), sharex=True, sharey=True, squeeze=False)
        fig.tight_layout()

        for i, idx in enumerate(img_idxs):
            row = int(i / nway)
            col = int(i % nway)

            img, label = dataset[idx]
            img = img.permute(1, 2, 0)
            img = destandardize(img, ds_name)

            axs[row, col].imshow(img)
            axs[row, col].get_xaxis().set_visible(False)
            axs[row, col].get_yaxis().set_visible(False)

        fname = os.path.join(outdir, f'img-{fidx}-acc-{acc}.png')
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

def plot_all_acc_hist(all_test_accs, outdir):

    all_accs = []

    for episode, epsval in all_test_accs.items():
        for label, labelvals in epsval.items():
            for batch, batchvals in labelvals.items():
                all_accs.append(batchvals['acc'])

    plt.figure()
    sns.histplot(data=np.array(all_accs), kde=True)
    plt.xlabel('Test Accuracy')
    plt.ylabel('Count')
    plt.title(f'Test accuracy')
    # plt.xlim((0, 100))
    plt.savefig(os.path.join(outdir, f'hist-all.png'), bbox_inches='tight')


def print_stats(all_test_accs):

    print('='*100)
    print(' '*40 + 'Stats')
    print('='*100)

    for episode, epsval in all_test_accs.items():
        eps_accs = []
        for label, labelvals in epsval.items():
            accs = [x['acc'] for x in labelvals.values()]
            eps_accs.extend(accs)
            print(f'\t\t Search label {label}. Min acc: {min(accs)}. Max accs: {max(accs)}')
        
        print(f'\tEpisode: {episode}. Min acc: {min(eps_accs)}. Max acc: {max(eps_accs)}')
        print('-'*100)


def verify_min_acc(embedding_net, cls_head, imgidxs, minacc, dloader, X_test, Y_test, opt):

    labels = {}
    for dl_label, dl_idxs in dloader.label2ind.items():
        for idx in imgidxs:
            if idx in dl_idxs:
                labels[idx] = dl_label
    
    examples = [(idx, labels[idx]) for idx in imgidxs]
    random.shuffle(examples)

    X_train, y_train = dloader.createExamplesTensorData(examples)

    X_train = X_train.cuda()
    y_train = y_train.cuda()

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    with torch.no_grad():

        emb_train = embedding_net(X_train)
        emb_train = emb_train.reshape(1, n_train, -1)

        emb_test = embedding_net(X_test)
        emb_test = emb_test.reshape(1, n_test, -1)

        if opt.head == 'SVM':
            logits = cls_head(emb_test, emb_train, y_train, opt.way, opt.shot, maxIter=3)
        else:
            logits = cls_head(emb_test, emb_train, y_train, opt.way, opt.shot)
    
        acc = count_accuracy(logits.reshape(-1, opt.way), y_test.reshape(-1))
        assert acc == minacc, f'Computed accuracy: {acc}. Minimum accuracy: {minacc}'
        print('Acc verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./experiments/exp_1/best_model.pth',
                            help='path of the checkpoint file')
    parser.add_argument('--ntasks', type=int, default=1, help="Number of different tasks to test on")
    parser.add_argument('--start_taskidx', type=int, default=0)
    parser.add_argument('--n_adv_rounds', type=int, default=1, help="Number of adversarial rounds")
    parser.add_argument('--way', type=int, default=5,
                            help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--phase', type=str, default='test', help="Validate adversarial accuracy on Training/Validation/Test phase")
    parser.add_argument('--minmax_acc', type=str, default="min", help="Optimize for minimum/maximum accuracy")
    parser.add_argument('--subsample_imgs', type=int, default=-1, help="Subsample images to search over")

    opt = parser.parse_args()
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)

    (dataset_test, data_loader) = get_dataset(opt)

    task_minaccs = []

    for taskidx in tqdm(range(opt.start_taskidx, opt.start_taskidx + opt.ntasks), desc='Tasks'):

        seed = opt.seed + taskidx
        random.seed(seed)
        np.random.seed(seed)

        if opt.phase == 'test':
            novel_cats_original = random.sample(list(dataset_test.labelIds_novel), opt.way)
        elif opt.phase == 'train':
            novel_cats_original = random.sample(list(dataset_test.labelIds_base), opt.way)
        elif opt.phase == 'val':
            novel_cats_original = random.sample(list(dataset_test.labelIds_novel), opt.way)

        
        dloader_test = data_loader(
            dataset=dataset_test,
            novel_categories=novel_cats_original,
            nTest=200,
            nShot=opt.shot,
            subsample_imgs=opt.subsample_imgs
        )

        print(novel_cats_original)
        
        # Randomly sample n_shot datapoint for each category
        fixed_imgIds = {}
        n_train = 0
        for label in dloader_test.categories:
            n_train = max(n_train, len(dloader_test.label2ind_train[label]))
            fixed_imgIds[label] = random.sample(list(dloader_test.label2ind_train[label]), opt.shot)

        set_gpu(opt.gpu)
        
        # log_file_path = os.path.join(os.path.dirname(opt.load), "test_log_exp.txt")
        worstacc_logpath = os.path.join(opt.outdir, f'worst-accs.txt')
        log_file_path = os.path.join(opt.outdir, f'test_log_exp-{taskidx}.txt')
        log(log_file_path, str(vars(opt)))

        # Define the models
        (embedding_net, cls_head) = get_model(opt)
        
        # Load saved model checkpoints
        saved_models = torch.load(opt.load)
        embedding_net.load_state_dict(saved_models['embedding'])
        embedding_net.eval()
        cls_head.load_state_dict(saved_models['head'])
        cls_head.eval()

        X_test, y_test = dloader_test.X_test, dloader_test.y_test
        X_test = X_test.cuda()
        y_test = y_test.cuda()

        all_accs = np.full(shape=(opt.n_adv_rounds, opt.way, opt.shot, n_train), fill_value=np.NaN)
        all_img_idxs = []
        worst_acc = 0.0 if opt.minmax_acc == 'max' else float("inf")
        worst_img_idxs = []
        idxs2accs = {}
        # task_log_step = 0

        for episode_idx in tqdm(range(opt.n_adv_rounds), desc='Episode', leave=False):
            for label_idx, label in enumerate(tqdm(dloader_test.categories, desc='Labels loop', leave=False)):
                for sample_idx in tqdm(range(opt.shot), desc="Sample index (shot)", leave=False):

                    fixed_imgIds_copy = copy.deepcopy(fixed_imgIds)
                    fixed_imgIds_copy[label].pop(sample_idx)

                    for batch_idx, batch in enumerate(tqdm(dloader_test.sample_episode_new(fixed_imgIds_copy, label), leave=False, desc="Batch loop")):

                        X_train, y_train, img_idx = batch
                        X_train = X_train.cuda()
                        y_train = y_train.cuda()

                        n_train = X_train.shape[0]
                        n_test = X_test.shape[0]

                        with torch.no_grad():
                            emb_train = embedding_net(X_train)
                            emb_train = emb_train.reshape(1, n_train, -1)

                            emb_test = embedding_net(X_test)
                            emb_test = emb_test.reshape(1, n_test, -1)

                            if opt.head == 'SVM':
                                logits = cls_head(emb_test, emb_train, y_train, opt.way, opt.shot, maxIter=3)
                            else:
                                logits = cls_head(emb_test, emb_train, y_train, opt.way, opt.shot)
                        
                            acc = count_accuracy(logits.reshape(-1, opt.way), y_test.reshape(-1))
                            
                            all_accs[episode_idx, label_idx, sample_idx, batch_idx] = acc.item()

                        tmp_idx = (episode_idx, label_idx, sample_idx, batch_idx)
                        tmp_img_dict = copy.deepcopy(fixed_imgIds_copy)
                        tmp_img_dict[label].append(img_idx)
                        tmp_imgids = list(tmp_img_dict.values())
                        tmp_imgids = [item for sublist in tmp_imgids for item in sublist]
                        all_img_idxs.append((tmp_idx, tmp_imgids))

                        if tuple(tmp_imgids) not in idxs2accs:
                            idxs2accs[tuple(tmp_imgids)] = acc.item()

                        if (opt.minmax_acc == 'min' and acc.item() < worst_acc) or (opt.minmax_acc == 'max' and acc.item() > worst_acc):
                            worst_acc = acc.item()
                            worst_img_idxs = all_img_idxs[-1]
                            fixed_imgIds[label][sample_idx] = img_idx

                        torch.cuda.empty_cache()
                    
                    np.save(os.path.join(opt.outdir, f'all-accs-{taskidx}'), all_accs)
                    np.save(os.path.join(opt.outdir, f'all-img-idxs-{taskidx}'), all_img_idxs)

                    tmp = all_accs[episode_idx, label_idx, sample_idx]
                    msgstr = f'Episode: {episode_idx}, Label: {label}, Sample idx: {sample_idx}, ' + \
                            f'Worst: {worst_acc}, Min: {np.nanmin(tmp)}, Avg: {np.nanmean(tmp)}, Max: {np.nanmax(tmp)}'
                    log(log_file_path, msgstr)
                
                log(log_file_path, '\n' + '*'*100 + '\n')
            
            log(log_file_path, str(worst_img_idxs))
            log(log_file_path, '\n\n' + '='*100 + '\n\n')

        if opt.minmax_acc == 'min':
            assert np.nanmin(all_accs) == worst_acc, f"Min acc: {all_accs.min()}. Worst acc: {worst_acc}"
        elif opt.minmax_acc == 'max':
            assert np.nanmax(all_accs) == worst_acc, f'Max acc: {all_accs.max()}. Worst acc: {worst_acc}'


        verify_min_acc(embedding_net, cls_head, worst_img_idxs[1], worst_acc, dloader_test, X_test, y_test, opt)

        log(log_file_path, f'Final images: {fixed_imgIds} \n Worst images: {worst_img_idxs}')
        img_outdir = os.path.join(opt.outdir, f'images-{taskidx}')
        # print_stats(all_test_accs)

        log(worstacc_logpath, f'Task: {taskidx}, Worst acc: {worst_acc}')

        task_minaccs.append(worst_acc)

        with open(os.path.join(opt.outdir, f'plt-allaccs-hist-{taskidx}.pkl'), 'wb') as f:
            pickle.dump(idxs2accs, f)

        # try:
        #     plot_images(all_accs, all_img_idxs, dataset_test, opt.dataset, img_outdir, opt.shot, opt.way)
        # except Exception as err:
        #     print(err)

    np.save(os.path.join(opt.outdir, f'task-minaccs-{opt.start_taskidx}'), task_minaccs)
    overall_logpath = os.path.join(opt.outdir, f'test_log_overall.txt')
    log(overall_logpath, f'Avg min accs: {np.average(task_minaccs)}')
    log(overall_logpath, f'Stddev min accs: {np.std(task_minaccs)}')
    log(overall_logpath, '-'*100)

    log(overall_logpath, f'All task min accs: {task_minaccs}')
