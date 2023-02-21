import torch
import numpy as np
import os
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x


class JoinedRestSegBatchDataset(object):

    def __init__(self, vol_ds, batch_size, epoch_len=250, p_fg=0,
                 mn_fg=3, store_coords_in_ram=True, store_data_in_ram=False,
                 n_max_volumes=None, memmap='r', return_fp16=True,
                 fbp_key='fbp', image_key='image', label_key='label'):
        self.vol_ds = vol_ds
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.p_fg = p_fg
        self.mn_fg = mn_fg
        self.store_coords_in_ram = store_coords_in_ram
        self.store_data_in_ram = store_data_in_ram
        self.return_fp16 = return_fp16
        self.n_max_volumes = len(self.vol_ds) if n_max_volumes is None else n_max_volumes
        self.memmap = memmap
        self.image_key = image_key
        self.label_key = label_key
        self.fbp_key = fbp_key

        if self.store_data_in_ram:
            print('Store data in RAM.\n')
            self.data = []
            for ind in tqdm(range(self.n_max_volumes)):
                path_dict = self.vol_ds.path_dicts[ind]
                seg = np.load(path_dict[self.label_key])
                im = np.load(path_dict[self.image_key])
                fbp = np.load(path_dict[self.fbp_key])
                if self.return_fp16:
                    im = im.astype(np.float16)
                    fbp = fbp.astype(np.float16)
                self.data.append((fbp, im, seg))
        self.dtype = np.float16 if self.return_fp16 else np.float32

        # store coords in ram
        if self.store_coords_in_ram:
            print('Precomputing foreground coordinates to store them in RAM')
            self.coords_list = []
            for ind in tqdm(range(self.n_max_volumes)):
                if self.store_data_in_ram:
                    seg = self.data[ind][2]
                else:
                    data_dict = self.vol_ds[ind]
                    seg = data_dict['label']
                if seg.max() > 0:
                    coords = np.stack(np.where(np.sum(seg, (1, 2)) > 0)[0]).astype(np.int16)
                else:
                    coords = np.array([])
                self.coords_list.append(coords)
            print('Done')
        else:
            self.bias_coords_fol = os.path.join(self.vol_ds.preprocessed_path, 'bias_coordinates_z')
            if not os.path.exists(self.bias_coords_fol):
                os.mkdir(self.bias_coords_fol)

            # now we check if come cases are missing in the folder
            print('Checking if all bias coordinates are stored in '+self.bias_coords_fol)
            for d in self.vol_ds.path_dicts:
                case = os.path.basename(d[self.label_key])
                if case not in os.listdir(self.bias_coords_fol):
                    lb = np.load(d[self.label_key])
                    coords = np.stack(np.where(np.sum(lb, (1, 2)) > 0)[0])
                    coords = coords.astype(np.int16)
                    np.save(os.path.join(self.bias_coords_fol, case), coords)

    def _get_volume_tuple(self, ind=None):

        if ind is None:
            ind = np.random.randint(self.n_max_volumes)
        if self.store_data_in_ram:
            fbp, im, seg = self.data[ind]
        else:
            path_dict = self.vol_ds.path_dicts[ind]
            fbp = np.load(path_dict[self.fbp_key], 'r')
            im = np.load(path_dict[self.image_key], 'r')
            seg = np.load(path_dict[self.label_key], 'r')
        return fbp, im, seg

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index=None):
        # makes a new batch and stores it
        # we're doing this so that the __getitem__ function can return samples
        # instead of batches
        fbps = []
        ims = []
        segs = []

        # draw the number of patches with forces fg in the batch
        n_fg_samples = self.mn_fg + np.sum([np.random.rand() < self.p_fg
                                            for _ in range(self.batch_size -
                                                           self.mn_fg)])

        for b in range(self.batch_size):

            # draw random index
            ind = np.random.randint(self.n_max_volumes)

            # load the memory maps of the data
            fbp, im, seg = self._get_volume_tuple(ind)

            # how many fg samples do we alreay have in the batch?
            k_fg_samples = np.sum([np.max(samp > 0) for samp in segs])
            if k_fg_samples < n_fg_samples:
                # if we're not there let's choose a center coordinate
                # that contains fg
                if self.store_coords_in_ram:
                    coords = self.coords_list[ind]
                else:
                    # or not!
                    case = os.path.basename(self.vol_ds.path_dicts[ind][self.label_key])
                    coords = np.load(os.path.join(self.bias_coords_fol, case))
                n_coords = len(coords)
                if n_coords > 0:
                    zcoord = coords[np.random.randint(n_coords)]
                else:
                    # random coordinate
                    zcoord = np.random.randint(seg.shape[0])
            else:
                # random coordinate
                zcoord = np.random.randint(seg.shape[0])
            # now get the cropped and padded sample
            fbps.append(fbp[np.newaxis, zcoord])
            ims.append(im[np.newaxis, zcoord])
            segs.append(seg[np.newaxis, zcoord])

        # stack up in first dim except for the segmentations as they have
        # different resolutions
        batch = (np.stack(fbps).astype(self.dtype),
                 np.stack(ims).astype(self.dtype),
                 np.stack(segs).astype(self.dtype))

        return batch

def JoinedRestSegDataloader(vol_ds, batch_size, num_workers=None,
                            pin_memory=True, epoch_len=250, p_fg=0,
                            mn_fg=3, store_coords_in_ram=True, memmap='r',
                            store_data_in_ram=False, n_max_volumes=None,
                            return_fp16=True):
    dataset = JoinedRestSegBatchDataset(vol_ds, batch_size,
                                        epoch_len=epoch_len,
                                        p_fg=p_fg, mn_fg=mn_fg,
                                        store_coords_in_ram=store_coords_in_ram,
                                        store_data_in_ram=store_data_in_ram,
                                        n_max_volumes=n_max_volumes,
                                        return_fp16=return_fp16)
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 8
    worker_init_fn = lambda _: np.random.seed()
    return torch.utils.data.DataLoader(dataset,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       worker_init_fn=worker_init_fn)
