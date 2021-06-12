import numpy as np
from torch.utils.data import DataLoader, Dataset

from datasets.dataset_utils import dataset_with_img_rays_together


class FakeDataset(Dataset):
    def __init__(self, num_imgs, num_rays_per_img):
        self.items = []
        for i in range(num_imgs):
            t = [i] * num_rays_per_img
            r = range(num_rays_per_img)
            im_rays = np.array(list(zip(t, r)))
            self.items.append(im_rays)
        self.items = np.concatenate(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    img_wh = (30, 23)
    batch_size = 20
    ds = FakeDataset(20, img_wh[0] * img_wh[1])
    ds2 = dataset_with_img_rays_together(ds, img_wh, batch_size=batch_size, num_imgs_in_batch=5)
    dl = DataLoader(
        ds2,
        shuffle=False,
        num_workers=1,
        batch_size=batch_size,
    )

    for batch_idx, data in enumerate(dl):
        np = np.array(zip(*data))
        print(f"Batch idx {batch_idx}")
        print(data)
        break
