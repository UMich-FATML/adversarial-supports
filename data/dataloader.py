from __future__ import print_function

import os
import os.path
import numpy as np
import random
import pickle
import json
import math
from copy import deepcopy

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt

from PIL import Image
from PIL import ImageEnhance

from pdb import set_trace as breakpoint


def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy


# DataLoader for adversarial training of model
# Takes in the current model to select the worst-possible
#  examples to use for training

class AdversarialFewShotDataloader():
    def __init__(self,
                 dataset,
                 nKnovel=5, # number of novel categories.
                 nKbase=-1, # number of base categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=15*5, # number of test examples for all the novel categories.
                 nTestBase=15*5, # number of test examples for all the base categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=4,
                 epoch_size=2000, # number of batches per epoch.
                 ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase=='train'
                                else self.dataset.num_cats_novel)
        assert(nKnovel >= 0 and nKnovel < max_possible_nKnovel)
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase=='train' and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert(nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert(cat_id in self.dataset.label2ind)
        assert(len(self.dataset.label2ind[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def convertDict2Tensor(self, img_ids_dict):

        exemplars = []
        for category, imgs in img_ids_dict.items():
            exemplars += [(img_id, category) for img_id in imgs]

        random.shuffle(exemplars)
        return self.createExamplesTensorData(exemplars)

    def adaptModelAndComputeAccuracy(self, data_support, data_query, model, n_way, n_shot):
        
        X_support, Y_support = data_support
        X_query, Y_query = data_query
        X_support = X_support.unsqueeze(0)      # add 0th dimension (=1) for batch size
        X_query = X_query.unsqueeze(0)          # (1, n, 3, 32, 32)   

        # X_support.cuda()
        # Y_support.cuda()
        # X_query.cuda()
        # Y_query.cuda()
        
        embedding_net, cls_head = model

        n_support = X_support.shape[1]
        n_query = X_query.shape[1]


        with torch.no_grad():
            emb_support = embedding_net(X_support.reshape([-1] + list(X_support.shape[-3:])))
            emb_support = emb_support.reshape(1, n_support, -1)

            emb_query = embedding_net(X_query.reshape([-1] + list(X_query.shape[-3:])))
            emb_query = emb_query.reshape(1, n_query, -1)

            logit_query = cls_head(emb_query, emb_support, Y_support, n_way, n_shot)
            acc = count_accuracy(logit_query.reshape(-1, n_way), Y_query.reshape(-1))
            acc = acc.item()
        
        return acc

    def sampleAdversarialImageIdsFrom(self, categories, model, eval_sample_size, exemplar_sample_size, nKbase, TBase):

        # Sample the evaluation set
        Tnovel = []
        for cat_idx, cat in enumerate(categories):
            imd_ids = self.sampleImageIdsFrom(cat, sample_size=eval_sample_size)
            Tnovel += [(img_id, nKbase+cat_idx) for img_id in imd_ids]

        Test = TBase + Tnovel

        X_query, Y_query = self.createExamplesTensorData(Test)

        # Randomly sample `sample_size` images
        worst_acc = float("inf")
        adversarial_data = {}
        n_way = len(categories)
        n_shot = exemplar_sample_size

        for cat_idx, cat in enumerate(categories):
            img_ids = self.sampleImageIdsFrom(cat, exemplar_sample_size)
            adversarial_data[nKbase + cat_idx] = img_ids

        # Create adversarial data
        for sample_idx in range(exemplar_sample_size):
            for cat_idx, cat in enumerate(categories):
                adversarial_data_copy = deepcopy(adversarial_data)
                cat_imgs = self.dataset.label2ind[cat]

                for img_id in cat_imgs:
                    
                    # do not duplicate images while creating n-shot dataset
                    if img_id in adversarial_data_copy[nKbase + cat_idx]:
                        continue

                    adversarial_data_copy[nKbase + cat_idx][sample_idx] = img_id
                    X_adv, Y_adv = self.convertDict2Tensor(adversarial_data_copy)

                    # Compute accuracy and keep copy
                    acc = self.adaptModelAndComputeAccuracy(
                        (X_adv, Y_adv), (X_query, Y_query), model, n_way, n_shot
                    )

                    if acc < worst_acc:
                        worst_acc = acc
                        adversarial_data = deepcopy(adversarial_data_copy)
        
        # Convert dictionary data to list
        Exemplars = []
        for cat, imgs in adversarial_data.items():
            Exemplars += [(cat, img_id) for img_id in imgs]
        
        return Tnovel, Exemplars


    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.

        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set=='base':
            labelIds = self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.

        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories

        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        if self.is_eval_mode:
            assert(nKnovel <= self.dataset.num_cats_novel)
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sampleCategories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        """
        Sample `nTestBase` number of images from the `Kbase` categories.

        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            nTestBase: the total number of images that will be sampled.

        Returns:
            Tbase: a list of length `nTestBase` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `nTestBase`.
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert(len(Tbase) == nTestBase)

        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase, model, TBase):
        """Samples train and test examples of the novel categories.

        Args:
    	    Knovel: a list with the ids of the novel categories.
            nTestNovel: the total number of test images that will be sampled
                from all the novel categories.
            nExemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.

        Returns:
            Tnovel: a list of length `nTestNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        if model is None:
            # No model provided. Randomly sample data points

            for Knovel_idx in range(len(Knovel)):
                imd_ids = self.sampleImageIdsFrom(
                    Knovel[Knovel_idx],
                    sample_size=(nEvalExamplesPerClass + nExemplars))
                    
                imds_tnovel = imd_ids[:nEvalExamplesPerClass]
                imds_ememplars = imd_ids[nEvalExamplesPerClass:]

                Tnovel += [(img_id, nKbase+Knovel_idx) for img_id in imds_tnovel]
                Exemplars += [(img_id, nKbase+Knovel_idx) for img_id in imds_ememplars]
        else:
            # Model is provided. Sample adversarial data points
            Tnovel, Exemplars = self.sampleAdversarialImageIdsFrom(
                Knovel, model, nEvalExamplesPerClass, nExemplars, nKbase, TBase
            )

        assert(len(Tnovel) == nTestNovel)
        assert(len(Exemplars) == len(Knovel) * nExemplars)
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self, model):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase, model, Tbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def get_iterator(self, epoch=0, model=None):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            # global model
            # `model` comes from the passed argument
            Exemplars, Test, Kall, nKbase = self.sample_episode(model)
            Xt, Yt = self.createExamplesTensorData(Test)
            Kall = torch.LongTensor(Kall)
            if len(Exemplars) > 0:
                Xe, Ye = self.createExamplesTensorData(Exemplars)
                return Xe, Ye, Xt, Yt, Kall, nKbase
            else:
                return Xt, Yt, Kall, nKbase

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

    def __call__(self, model, epoch=0):
        return self.get_iterator(model, epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)

