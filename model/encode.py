from typing import List, Tuple
import torch
import random


import preprocessing.token
import model.autoencoder


def encode(autoencoder: torch.nn.Module,
           songs: List[Tuple[str, float, List[Tuple[int, int]]]]
           ) -> List[List[torch.tensor]]:
    """
    Encodes all songs and for each songs, returns a list of codes. Songs are
    padded with zero tokens so that their lengths become a multiple of the
    size of a sequence.

    autoencoder: an instance of a class defined in model.autoencoder
    songs: list of songs
    """
    nsongs = len(songs)

    # grab training parameters
    import model.train
    seq_len = model.train.seq_len
    batch_size = model.train.batch_size
    device = model.train.device
    ign_ind = preprocessing.token.tok2int((0, 0))
    autoencoder.to(device)
    autoencoder.eval()

    # this is how we turn each song into tensors of size seq_len x batch_size
    def make_ten(k):  # k: song number
        _, _, toks = songs[k]
        length = ((len(toks)+seq_len-1)//seq_len)*seq_len
        base = 0
        while base < len(toks):
            b = min(batch_size, ((length-base)+seq_len-1)//seq_len)
            t = torch.empty(seq_len, b, dtype=torch.long, device=device)
            for i in range(seq_len):
                for j in range(b):
                    n = base + j*seq_len+i
                    if n < len(toks):
                        t[i, j] = preprocessing.token.tok2int(toks[n])
                    else:
                        t[i, j] = ign_ind
            yield t
            base += seq_len * batch_size
    with torch.no_grad():
        codes = []  # all the encoded songs
        i=0
        for k in range(nsongs):
            chunks = []  # all code vectors
            i+=1
            for t in make_ten(k):
                b = t.size(1)  # may be smaller than batch_size
                c = autoencoder.encode(t)
                chunks += [c[:, j].flatten() for j in range(b)]  # append
            codes.append(chunks)
    return codes
