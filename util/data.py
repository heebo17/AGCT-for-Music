import torch
import random

import preprocessing.token


class dataset(torch.utils.data.Dataset):
    """
    Imprementation of torch data loader
    https://pytorch.org/docs/stable/data.html
    """
    def __init__(self, songs, seq_len, device):
        """
        songs: list of streams of tokens
        seq_len: windows size
        device: device to create tensors on
        """
        super(dataset, self).__init__()
        self.seq_len = seq_len
        self.nsongs = len(songs)
        self.lengths = []
        self.song = []
        for _, _, toks in songs:
            self.lengths.append(len(toks))
            self.song.append(
                torch.tensor(
                    [preprocessing.token.tok2int(t) for t in toks],
                    device=device,
                    dtype=torch.long
                    )
                )
        self.count = sum(l-seq_len+1 for l in self.lengths)

    def __len__(self):
        return self.count

    def __getitem__(self, t):
        s = t[0]
        offset = t[1]
        return self.song[s][offset:offset+self.seq_len]


class sampler(torch.utils.data.Sampler):
    """
    Imprementation of torch data loader
    https://pytorch.org/docs/stable/data.html
    """
    def __init__(self, dataset):
        """
        dataset: instance of the above class
        """
        super(sampler, self).__init__(dataset)
        self.keys = [
            (s, offset) for s in range(dataset.nsongs)
            for offset in range(len(dataset.song[s])-dataset.seq_len+1)
            ]
        random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        yield from self.keys
        random.shuffle(self.keys)


def collate_fn(l):
    # for GRU accept input of size: seq_len x batch_size
    return torch.stack(l, dim=1)
