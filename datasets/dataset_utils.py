from typing import List
import torch
# pytorch-lightning
from torch.utils.data import ConcatDataset, Subset, random_split


def dataset_with_img_rays_together(dataset, img_wh, batch_size, num_imgs_in_batch) -> ConcatDataset:
    """
    Ensure that each batch contains rays sampled from same image.
    See test_img_rays_together.py for example.
    Args:
        dataset: Actual dataset.
        img_wh: Image width and height.
        batch_size: Batch size to use i.e. number of rays in a batch.
        num_imgs_in_batch: Number of different images to use in a batch. e.g. if batch size is 1024 and
        num_imgs_in_batch is 2 -> 512 rays from 2 images will end up in a batch.

    Returns:
        Ordered ConcatDataset (do not shuffle!)
    """
    # Ensure that each batch contains rays sampled from same img

    num_rays = len(dataset)  # Total num of rays
    num_rays_per_img = img_wh[0] * img_wh[1]
    assert num_rays % num_rays_per_img == 0  # sanity check
    num_imgs = num_rays // num_rays_per_img  # Num images in dataset

    # arg check
    assert batch_size % num_imgs_in_batch == 0

    # chunk size
    # chunk = set of rays from an image inside a batch i.e. subset of batch
    # Chunks build the batches i.e. if num_imgs_in_batch = 2, there will be 2 chunks from two images in a batch
    num_rays_per_img_per_batch = batch_size // num_imgs_in_batch

    # num chunks per image
    num_full_chunks_per_img, last_chunk_size = divmod(num_rays_per_img, num_rays_per_img_per_batch)

    # We need to split the rays of an image into chunks
    # chunk sizes per image
    chunk_sizes_per_img = [num_rays_per_img_per_batch] * num_full_chunks_per_img
    if last_chunk_size > 0:  # leftover chunk
        chunk_sizes_per_img.append(last_chunk_size)
    assert sum(chunk_sizes_per_img) == num_rays_per_img

    subsets: List[List[Subset]] = []  # List of list of subsets
    for i in range(num_imgs):
        offset = i * num_rays_per_img
        # Take rays from a single image
        img_subset = Subset(dataset, indices=range(offset, offset + num_rays_per_img))
        # Random split rays from the single image
        # list of subsets
        random_subsets_ls = random_split(img_subset, lengths=chunk_sizes_per_img)
        subsets.append(random_subsets_ls)

    # Randomly sample chunks from images
    perm_idxs = torch.arange(num_imgs).repeat_interleave(num_full_chunks_per_img)
    # e.g. [0,0,0,0,1,1,1,1,...]
    perm_idxs = perm_idxs[torch.randperm(len(perm_idxs))]
    # e.g. [1,3,0,2,1,...]
    subsets_ordered = []
    for perm_idx in perm_idxs:
        # pop subset from the image with perm_idx
        subsets_ordered.append(subsets[perm_idx].pop(0))
    if last_chunk_size > 0:
        for perm_idx in torch.randperm(num_imgs):
            subsets_ordered.append(subsets[perm_idx].pop(0))
    # Chain the subsets one after another
    return ConcatDataset(subsets_ordered)


