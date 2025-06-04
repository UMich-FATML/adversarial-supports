import numpy as np
import torch

def test_embed_query(embed, batchsize, nquery):
    embed_reshape = embed.reshape(batchsize, nquery, -1)

    for i in range(embed.shape[0]):
        idx_0 = int(i / nquery)
        idx_1 = int(i % nquery)

        assert torch.all(torch.eq(embed_reshape[idx_0, idx_1, :], embed[i]))


def test_embed_support(embed, batchsize, nWay, subsample_imgs):

    embed_reshape = embed.reshape(batchsize, nWay, subsample_imgs, -1)

    for i in range(embed.shape[0]):
        idx_2 = int(i % subsample_imgs)
        rem = int(i / subsample_imgs)
        idx_1 = int(rem % nWay)
        idx_0 = int(rem / nWay)

        assert np.sum(embed_reshape[idx_0, idx_1, idx_2, :] != embed[i]) == 0


def test_image_reshape(imgs):
    
    imgs_reshape = imgs.reshape([-1] + list(imgs.shape)[-3:])
    bs, nway, samples = list(imgs.shape)[:-3]    # (bs, nway, sample_count)

    for i in range(imgs_reshape.shape[0]):
        idx_2 = int(i % samples)
        rem = int(i / samples)
        idx_1 = int(rem % nway)
        idx_0 = int(rem / nway)

        assert np.sum(
            imgs_reshape[i] != imgs[idx_0, idx_1, idx_2]
        ) == 0


def test_embeddings_by_idx(x_emb, y_emb):

    batch_size, nway, nshot, embed_dim = x_emb.shape
    x_emb_reshape = x_emb.reshape([batch_size, -1, embed_dim])
    y_emb_reshape = y_emb.reshape([batch_size, -1])

    for i in range(batch_size):
        for j in range(nway * nshot):

            idx_0 = int(j / nshot)
            idx_1 = int(j % nshot)

            assert np.sum(
                x_emb[i, idx_0, idx_1] != x_emb_reshape[i, j]
            ) == 0
            

def test_embeddings_idxs(embeddings, extracted_embeddings, idxs):
    
    batch_size, nway, nimgs, d = embeddings.shape
    nshot = extracted_embeddings.shape[2]

    for i in range(batch_size):
        for j in range(nway):
            for k in range(nshot):
                idx = int(idxs[i, j, k])
                assert np.sum(
                    embeddings[i, j, idx] != extracted_embeddings[i, j, k]
                ) == 0