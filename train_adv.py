# -*- coding: utf-8 -*-

import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from copy import deepcopy
import cProfile
import matplotlib.pyplot as plt

from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12

from utils import set_gpu, Timer, count_accuracy, check_dir, log, count_batched_accuracy
from utils_test import test_embed_query, test_embed_support, test_image_reshape, test_embeddings_by_idx, test_embeddings_idxs


_TEST_RESHAPE = False
_PROFILE_CODE = False


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

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
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_train = FC100(phase='train')
        dataset_val = FC100(phase='val')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, data_loader)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=1, help="Initial epoch number")
    parser.add_argument('--num-epoch', type=int, default=60,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--subsample_imgs', type=int, default=50, help="No. of images to subsample and search from")
    parser.add_argument('--load_model', type=str, default=None, help="Model path to load and warm-start")
    parser.add_argument('--fix_cats', type=int, default=0, help="Fix categories and train")
    parser.add_argument('--attack_rounds', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1, help="initial learning rate")
    
    return parser

################################################################
########        Adversarial data generation section     ########
################################################################

def plot_imgs(imgs, fname):
    nway, nshot = imgs.shape[:2]

    mean_pix = torch.Tensor([x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]])
    std_pix = torch.Tensor([x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]])
    mean_pix = mean_pix.view(-1, 1, 1)
    std_pix = std_pix.view(-1, 1, 1)

    fig, axs = plt.subplots(nway, nshot, figsize=(nshot*8, nway*8), squeeze=False)

    for i in range(nway):
        for j in range(nshot):
            axs[i][j].imshow(imgs[i, j].mul(std_pix).add(mean_pix).permute(1, 2, 0))

    plt.savefig(f'./{fname}.png', bbox_inches='tight')


def get_imgs_by_idxs_prefetched(data_imgs, img_idxs, nShot):

    shape = list(data_imgs.shape)
    shape[2] = nShot
    data = data_imgs.new_full(size=shape, fill_value=0.0)   # (batchsize, nway, nshot, datum_shape)

    for batchidx in range(shape[0]):
        for clsidx in range(shape[1]):
            idxs = data_imgs.new_tensor(data=img_idxs[batchidx, clsidx], dtype=torch.long)
            data[batchidx, clsidx] = torch.index_select(data_imgs[batchidx, clsidx], dim=0, index=idxs)

    return data


@torch.no_grad()
def subsample_images_from_support_data(dataset, Kall, batch_size, nWay, subsample_imgs):
    """
    Subsamples images from support data
    This subsampled space is used to search for adversarial examples

    Returns:
        1. data_idxs: (batch_size, nWay, subsample_imgs)
        2. labels_idxs: (batch_size, nWay, subsample_imgs)
        3. datum_shape: Image shape (channels, height, width)
    """

    # Holds subsampled set of image indices. 
    # Shape: (batch_size, n_way, n_subsample)    ->   (8, 5, 50)
    data_idxs = np.full(shape=(batch_size, nWay, subsample_imgs), fill_value=np.NaN, dtype=np.int)
    labels_idxs = np.full(shape=(batch_size, nWay, subsample_imgs), fill_value=np.NaN, dtype=np.int)
    datum_shape = None

    # Subsample and populate data_idxs variable
    for batch_idx in range(batch_size):
        batch_labels = Kall[batch_idx].numpy()
        for label_idx, label in enumerate(batch_labels):
            idxs_sampled = random.sample(dataset.label2ind[label], subsample_imgs)
            data_idxs[batch_idx, label_idx, :] = idxs_sampled
            labels_idxs[batch_idx, label_idx, :] = label_idx

            if datum_shape is None:
                datum_shape = dataset[idxs_sampled[0]][0].shape

    labels_idxs = torch.LongTensor(labels_idxs).cuda()
    
    return data_idxs, labels_idxs, datum_shape


@torch.no_grad()
def get_embeddings_by_idxs(embeddings, labels, batch_size, imgidxs, embed_dim, nWay):

    # Get embeddings for these imgidxs_loop

    # batch_x_emb shape: (batchsize, nWay, nShot, embed_dim)
    # batch_y_emb shape: (batchsize, nWay, nShot)
    batch_x_emb = embeddings.new_full(size=(*imgidxs.shape, embed_dim), fill_value=float('nan'), dtype=torch.float32)
    batch_y_emb = embeddings.new_full(size=imgidxs.shape, fill_value=-1, dtype=torch.long)

    for i in range(batch_size):
        for j in range(nWay):
            idxs = torch.LongTensor(imgidxs[i, j]).to(embeddings.device)
            batch_x_emb[i, j] = torch.index_select(embeddings[i, j], dim=0, index=idxs)
            batch_y_emb[i, j] = torch.index_select(labels[i, j], dim=0, index=idxs)

    if _TEST_RESHAPE:
        test_embeddings_by_idx(batch_x_emb, batch_y_emb)
        test_embeddings_idxs(embeddings, batch_x_emb, imgidxs)
    
    batch_x_emb = batch_x_emb.reshape([batch_size, -1, embed_dim])
    batch_y_emb = batch_y_emb.reshape([batch_size, -1])

    # batch_x_emb shape: (batch_size, nWay*nShot, embed_dim)
    # batch_y_emb shape: (batch_size, nWay*nShot)
    
    return batch_x_emb, batch_y_emb


@torch.no_grad()
def get_embeddings_multidim(embeddings, labels, indices):
    
    embed_dim = embeddings.shape[-1]
    batchsize, nimgs, nway, nshot = indices.shape
    
    batch_x_emb = embeddings.new_full(size=(*indices.shape, embed_dim), fill_value=float('nan'))
    batch_y_emb = embeddings.new_full(size=indices.shape, fill_value=-1, dtype=torch.long)

    for i in range(batchsize):
        for j in range(nimgs):
            for k in range(nway):
                idxs = torch.LongTensor(indices[i, j, k]).to(embeddings.device)
                batch_x_emb[i, j, k] = torch.index_select(embeddings[i, k], dim=0, index=idxs)
                batch_y_emb[i, j, k] = torch.index_select(labels[i, k], dim=0, index=idxs)
    
    batch_x_emb = batch_x_emb.reshape([batchsize, nimgs, -1, embed_dim])
    batch_y_emb = batch_y_emb.reshape([batchsize, nimgs, -1])

    return batch_x_emb, batch_y_emb


def get_images_data_given_idxs(dataset, data_idxs, batch_size, nWay, datum_shape, reshape=True):

    sample_count = data_idxs.shape[-1]
    # Shape: (batch_size, n_way, sample_count, *img_shape)  -> (8, 5, 50, 3, 32, 32)
    data_imgs = np.full(shape=(batch_size, nWay, sample_count, *datum_shape), fill_value=np.NaN)
    
    # Get image data from subsampled idxs
    for batch_idx in range(batch_size):
        for label_idx in range(nWay):
            for sample_idx in range(sample_count):
                idx = data_idxs[batch_idx, label_idx, sample_idx]
                img = dataset[idx][0]
                data_imgs[batch_idx, label_idx, sample_idx] = img

    # Precompute support data embeddings to avoid recomputing at each forward pass
    if reshape:
        if _TEST_RESHAPE:
            test_image_reshape(data_imgs)
        
        data_imgs = data_imgs.reshape([-1] + list(data_imgs.shape)[-3:])
    
    data_imgs_tensor = torch.from_numpy(data_imgs).float()
    return data_imgs_tensor


@torch.no_grad()
def createAdversarialSupportData(model, dataset, Kall, nShot, nWay,
        data_query, labels_query, nKbase, attack_rounds, subsample_imgs=50):

    batch_size = Kall.shape[0]
    assert batch_size == data_query.shape[0]

    datum_shape = None
    n_query = data_query.shape[1]
    data_query = data_query.cuda()
    labels_query = labels_query.cuda()

    embedding_net, cls_head = model
    # _, _ = [x.eval() for x in (embedding_net, cls_head)]

    emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))

    if _TEST_RESHAPE:
        test_embed_query(emb_query, batch_size, n_query)

    emb_query = emb_query.reshape(batch_size, n_query, -1)      # verified correct reshape

    # Subsample data from support data to define search space where adversarial examples are searched from
    data_idxs, labels_idxs, datum_shape = subsample_images_from_support_data(dataset, Kall, batch_size, nWay, subsample_imgs)
    
    # Store image data for all subsampled images in an array
    # Shape: (batch_size, n_way, n_subsample, *img_shape)  -> (8, 5, 50, 3, 32, 32)
    data_imgs = get_images_data_given_idxs(dataset, data_idxs, batch_size, nWay, datum_shape)
    data_imgs = data_imgs.cuda()

    # embedding_net return (batch_idx * nWay * subsample_imgs, d)  -> (8*5*50, 2560)
    emb_imgs = embedding_net(data_imgs)

    data_imgs = data_imgs.reshape([batch_size, nWay, subsample_imgs] + list(data_imgs.shape)[-3:])

    if _TEST_RESHAPE:
        test_embed_support(emb_imgs, batch_size, nWay, subsample_imgs)

    # emb_images shape: (batch_idx, nWay, subsample_imgs, d)
    emb_imgs = emb_imgs.reshape(batch_size, nWay, subsample_imgs, -1)
    embed_dim = emb_imgs.shape[-1]

    # Randomly sample n_shot images for each of the n_way class and batch
    # Holds adversarial data images
    adv_data_idxs = np.full(shape=(batch_size, nWay, nShot), fill_value=np.NaN, dtype=np.int)
    for i in range(batch_size):
        for j in range(nWay):
            adv_data_idxs[i, j] = np.random.choice(subsample_imgs, size=nShot, replace=False)

    min_acc = 100 * torch.ones(batch_size)
    min_acc = min_acc.cuda()
    nimgs = subsample_imgs - nShot + 1

    # Stack query data to match new batchsize
    emb_query_stacked = np.copy(emb_query.cpu().numpy())                                              # (batchsize, nquery, embeddim)
    emb_query_stacked = np.expand_dims(emb_query_stacked, axis=1)                       # (batchsize, 1, nquery, embeddim)
    emb_query_stacked = np.broadcast_to(emb_query_stacked, shape=(batch_size, nimgs, n_query, embed_dim))
    emb_query_stacked = emb_query_stacked.copy()                                        # (batchsize, nimgs, nquery, embeddim)
    emb_query_stacked = emb_query_stacked.reshape([-1] + list(emb_query.shape)[-2:])    # (batchsize*nimgs, nquery, embeddim)
    emb_query_stacked = emb_query.new_tensor(data=emb_query_stacked)

    labels_query_stacked = np.copy(labels_query.cpu().numpy())
    labels_query_stacked = np.expand_dims(labels_query_stacked, axis=1)
    labels_query_stacked = np.broadcast_to(labels_query_stacked, shape=(batch_size, nimgs, n_query))
    labels_query_stacked = labels_query_stacked.copy()
    labels_query_stacked = labels_query_stacked.reshape([-1, n_query])
    labels_query_stacked = labels_query.new_tensor(data=labels_query_stacked)           # (batchsize*nimgs, nquery)

    for round in tqdm(range(attack_rounds), desc=f"Attack round. Idxs: {data_idxs[0, 0, 0:5]}", leave=False):
        with tqdm(range(nWay), desc="Class index", leave=False) as t:
            for clsidx in t:
                for sampleidx in tqdm(range(nShot), desc="Sample index", leave=False):

                    available_idxs = np.full(shape=(batch_size, subsample_imgs - nShot), fill_value=np.NaN)
                    for batchidx in range(batch_size):
                        available_idxs[batchidx, :] = np.setdiff1d(
                            np.arange(subsample_imgs), adv_data_idxs[batchidx, clsidx].flatten()
                        )
                        
                    batch_adv_dataidxs = np.copy(adv_data_idxs)     # (batchsize, nWay, nShot)
                    batch_adv_dataidxs = np.expand_dims(batch_adv_dataidxs, axis=1)     # (batchsize, 1, nWay, nShot)
                    batch_adv_dataidxs = np.broadcast_to(batch_adv_dataidxs, shape=(batch_size, nimgs, nWay, nShot))
                    batch_adv_dataidxs = batch_adv_dataidxs.copy()
                    
                    batch_adv_dataidxs[:, 1:, clsidx, sampleidx] = available_idxs

                    batch_x_emb, batch_y_emb = get_embeddings_multidim(emb_imgs, labels_idxs, batch_adv_dataidxs)

                    # Shape:
                    #   batch_x_emb:        (batchsize, nimgs, nway*nshot, emb_dim)
                    #   batch_y_emb:        (batchsize, nimgs, nway*nshot)
                    #   batch_adv_dataidxs: (batchsize, nimgs, nway, nshot)

                    ##########################################
                    ##############  Reshaping          #######
                    ##########################################
                    
                    # Shape:
                    #   batch_x_emb:        (batchsize*nimgs, nway*nshot, emb_dim)
                    #   batch_y_emb:        (batchsize*nimgs, nway*nshot)
                    #   batch_adv_dataidxs: (batchsize*nimgs, nway, nshot)

                    batch_x_emb = batch_x_emb.reshape([-1] + list(batch_x_emb.shape)[-2:])
                    batch_y_emb = batch_y_emb.reshape([-1] + list(batch_y_emb.shape)[-1:])
                    # batch_adv_dataidxs = batch_adv_dataidxs.reshape(-1, nWay, nShot)

                    # Get accuracy for each using classification head
                    logit_query = cls_head(emb_query_stacked, batch_x_emb, batch_y_emb, nWay, nShot)

                    batch_acc = count_batched_accuracy(logit_query, labels_query_stacked)

                    batch_acc = batch_acc.reshape([batch_size, nimgs])
                    min_batch_accs, min_batch_accidxs = torch.min(batch_acc, dim=1)
                    batch_mask = (min_batch_accs < min_acc)

                    for i in range(batch_size):
                        if not batch_mask[i]:
                            continue
                        adv_data_idxs[i] = batch_adv_dataidxs[i, min_batch_accidxs[i]]

                    min_acc = torch.min(min_acc, min_batch_accs)
                    t.set_postfix(min_acc=min_acc.mean().item())

    x_adv = get_imgs_by_idxs_prefetched(data_imgs, adv_data_idxs, nShot)
    x_adv = x_adv.view(batch_size, -1, *datum_shape).cuda()
    x_embed, y_adv = get_embeddings_by_idxs(emb_imgs, labels_idxs, batch_size, adv_data_idxs, embed_dim, nWay)

    # Verify computed minimum accuracy with adversarial images accuracy
    x_embed = x_embed.cuda()
    y_adv = y_adv.cuda()
    logit_query = cls_head(emb_query, x_embed, y_adv, nWay, nShot)
    batch_acc = count_batched_accuracy(logit_query, labels_query)
    batch_acc = batch_acc.cpu()

    try:
        assert (torch.eq(batch_acc, min_acc.cpu()).all().numpy() == True)
    except Exception as err:
        print('='*100)
        print('Assertion Exception occurred')
        print(f'Batch acc: {batch_acc}')
        print(f'Min acc: {min_acc.cpu()}')
        # raise err

    del x_embed
    torch.cuda.empty_cache()
    # _, _ = [x.train() for x in (embedding_net, cls_head)]

    # Permute datapoints
    for batchidx in range(batch_size):
        random.seed(batchidx)
        perm = torch.randperm(x_adv.shape[1], dtype=torch.long, device=x_adv.device)
        x_adv[batchidx, :] = x_adv[batchidx, perm]
        y_adv[batchidx, :] = y_adv[batchidx, perm]

    assert x_adv.shape[0] == batch_size
    assert x_adv.shape[1] == nShot * nWay
    assert y_adv.shape[0] == batch_size
    assert y_adv.shape[1] == nShot * nWay

    return x_adv, y_adv, min_acc

################################################################
#######        Adversarial data generation section ends   ######
################################################################

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    parser = get_parser()
    opt = parser.parse_args()
    
    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    set_seed(0)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 500, # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    print('LOG FILE: ', log_file_path)

    (embedding_net, cls_head) = get_model(opt)

    if opt.load_model is not None:
        saved_models = torch.load(opt.load_model)
        embedding_net.load_state_dict(saved_models['embedding'])
        cls_head.load_state_dict(saved_models['head'])
    
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, 
                                 {'params': cls_head.parameters()}], lr=opt.lr, momentum=0.9, \
                                          weight_decay=5e-4, nesterov=True)
    
    lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    adv_min_accs, train_min_accs = [], []
    
    for epoch in range(opt.start_epoch, opt.start_epoch + opt.num_epoch):
        # Train on the training split
        lr_scheduler.step()
        
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_losses = []

        adv_min_accs.append([])

        with tqdm(dloader_train(epoch), desc="Epochs") as t:
            for i, batch in enumerate(t, 1):
                # _, _ = [x.train() for x in (embedding_net, cls_head)]
                # data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
                
                # `Kall` contains the sampled class labels for the current iteration
                # Use `Kall` to sample adversarial examples

                set_seed(epoch * 500 + i)

                _, _, data_query, labels_query, Kall, nKbase = batch
                data_query = data_query.cuda()
                labels_query = labels_query.cuda()

                # Create adversarial data_support
                if _PROFILE_CODE:
                    pr = cProfile.Profile()
                    pr.enable()

                model = (embedding_net, cls_head)

                with torch.no_grad():
                    data_support, labels_support, batch_min_acc = createAdversarialSupportData(
                        model, dataset_train, Kall, opt.train_shot, opt.train_way, data_query, 
                        labels_query, nKbase=0, attack_rounds=opt.attack_rounds, subsample_imgs=opt.subsample_imgs)

                adv_min_accs[-1].append(batch_min_acc.cpu().numpy())
                
                if _PROFILE_CODE:
                    pr.disable()
                    pr.print_stats(sort="cumtime")

                data_support = data_support.cuda()
                labels_support = labels_support.cuda()

                train_n_support = opt.train_way * opt.train_shot
                train_n_query = opt.train_way * opt.train_query

                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
                
                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
                
                logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

                smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
                smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

                log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()
                
                acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))
                
                batch_acc = count_batched_accuracy(logit_query, labels_query)
                train_min_accs.append(batch_acc.cpu().numpy())
                
                train_accuracies.append(acc.item())
                train_losses.append(loss.item())

                t.set_postfix(acc=acc.item())

                if (i % 10 == 0) or (i == 1):
                    train_acc_avg = np.mean(np.array(train_accuracies))
                    train_min_acc = np.mean(np.mean(np.array(train_min_accs), axis=1))
                    adv_min_acc = np.mean(adv_min_accs[-1])
                    train_loss_avg = np.mean(np.array(train_losses))

                    log(log_file_path, '*'*100)

                    log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                                epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))
                    
                    log(log_file_path, '\t\t\t Avg minimum train acc: {:.2f} % ({:.2f} %)'.format(train_min_acc, batch_acc.mean().item()))
                    log(log_file_path, '\t\t\t Avg Adversarial min acc: {:.2f} % ({:.2f} %)'.format(adv_min_acc, batch_min_acc.mean().item()))
                    
                    np.save(os.path.join(opt.save_path, 'adv_min_accs'), adv_min_accs)
                    np.save(os.path.join(opt.save_path, 'train_min_accs'), train_min_accs)

                    torch.cuda.empty_cache()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []
        
        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())
            
        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        np.save(os.path.join(opt.save_path, 'adv_min_accs'), adv_min_accs)
        np.save(os.path.join(opt.save_path, 'train_min_accs'), train_min_accs)

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))

