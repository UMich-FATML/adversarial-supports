import numpy as np
import torch
import random


class AdversarialFewShotTestDataloader():
    def __init__(self,
                dataset,
                novel_categories,
                nTest,
                nShot,
                batch_size=1,
                num_workers=4,
                subsample_imgs=None
                ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        self.novel_categories = novel_categories
        self.nTest = nTest
        self.nShot = nShot
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fixed_label_idxs = {}
        self.subsample_imgs = subsample_imgs

        self.prepareDataSplits()

    def prepareDataSplits(self):

        self.label2ind = {
            i: self.dataset.label2ind[label] 
            for i, label in enumerate(self.novel_categories)
        }

        self.categories = list(self.label2ind.keys())
        self.label2ind_test = {}
        self.label2ind_train = {}

        for label, idxs in self.label2ind.items():
            idxs = np.random.permutation(idxs)
            self.label2ind_test[label] = idxs[:self.nTest]

            if self.subsample_imgs > 0:
                self.label2ind_train[label] = idxs[self.nTest : self.nTest + self.subsample_imgs]
            else:
                self.label2ind_train[label] = idxs[self.nTest:]

        self.X_test, self.y_test = self.createTestSet()

    def fixLabelIdxs(self, fixed_label_ind):
        self.fixed_label_idxs = fixed_label_ind

    def createTestSet(self):

        # Create test dataset tuple
        test_examples = []
        for label, idxs in self.label2ind_test.items():
            tmp = [(idx, label) for idx in idxs]
            test_examples.extend(tmp)

        X_test, y_test = self.createExamplesTensorData(test_examples)
        return X_test, y_test

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

    def sample_episode_new(self, fixed_imgs, sample_label):
        
        train_set = []
        for label, idxs in fixed_imgs.items():
            tmp = [(idx, label) for idx in idxs]
            train_set.extend(tmp)
        
        label_idxs = [
            idx for idx in self.label2ind_train[sample_label]
            if idx not in fixed_imgs.get(sample_label, [])
        ]

        for i, idx in enumerate(label_idxs):
            random.seed(i)
            tmpset = np.random.permutation(train_set + [(idx, sample_label)])
            X_train, y_train = self.createExamplesTensorData(tmpset)
            yield X_train, y_train, idx
