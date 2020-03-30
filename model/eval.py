from typing import List, Tuple
import io
import sys
import torch

import preprocessing.token
import model.autoencoder


def eval(autoencoder: torch.nn.Module,
         evalset: List[Tuple[str, float, List[Tuple[int, int]]]],
         logfile: io.TextIOBase=sys.stdout
         ) -> List[Tuple[str, float, List[Tuple[int, int]]]]:
    """
    Evaluate autoencoder on evalset.
    autoencoder: an instance of a class defined in model.autoencoder
    evalset: list of songs
    logfile: where to write some statistics (default: stdout)
    """
    nsongs = len(evalset)
    result = []  # all the predicted songs

    import model.train  # grab training parameters
    seq_len = model.train.seq_len
    batch_size = model.train.batch_size
    device = model.train.device
    ign_ind = preprocessing.token.tok2int((0, 0))
    loss_fn = torch.nn.NLLLoss(
        ignore_index=ign_ind,
        reduction="sum"  # Compute average manually
    )
    autoencoder.to(device)
    autoencoder.eval()

    # this is how we turn each song into tensors of size seq_len x batch_size
    def make_ten(k):  # k: song number
        _, _, toks = evalset[k]
        base = 0
        while base < len(toks):
            t = torch.empty(seq_len, batch_size,
                           dtype=torch.long,
                            device=device)
            for i in range(seq_len):
                for j in range(batch_size):
                    index = base + j*seq_len+i
                    if index < len(toks):
                        t[i, j] = preprocessing.token.tok2int(
                            toks[index])
                    else:
                        t[i, j] = ign_ind
            yield t
            base += seq_len * batch_size
    # this is how we turn a list of predictions back into as song:
    def make_song(k, pred):  # k: song number, pred: predictions
        toks = []
        length = len(evalset[k][2])
        for t in pred:
            for j in range(batch_size):
                for i in range(seq_len):
                    toks.append(preprocessing.token.int2tok(t[i, j].item()))
                    length -= 1
                    if length == 0:
                        return (evalset[k][0], evalset[k][1], toks)
    with torch.no_grad():
        ncorrect_tot = 0  # total stats
        loss_tot = 0.
        for k in range(nsongs):
            ncorrect_song = 0  # per song stats
            loss_song = 0.
            chunks = []
            # Disjoint windows for prediction
            for t in make_ten(k):
                p, pred = autoencoder(t, False)  # no tf
                chunks.append(pred)  # save predictions of current chunk
                # collect statisticts
                loss = loss_fn(
                    p.view(seq_len*batch_size, autoencoder.dict_size),
                    t.view(seq_len*batch_size)
                    )
                npredict = 0
                ncorrect = 0
                for i in range(seq_len):
                    for j in range(batch_size):
                        if t[i, j].item() is ign_ind:
                            continue
                        npredict += 1
                        if t[i, j] == pred[i, j]:
                            ncorrect += 1.
                # # print some live statistics
                # acc = ncorrect / npredict
                # loss_avg = loss.item() / npredict
                # logfile.write(f"  Song {k}/{nsongs}: "
                #               f"loss {loss_avg:.4f}, accuracy {acc:.4f}   \r")
                # logfile.flush()
                # update song statistics
                ncorrect_song += ncorrect
                loss_song += loss.item()
            # song finished
            result.append(make_song(k, chunks))
            # print statistics
            length = len(evalset[k][2])
            logfile.write(f" Song {k}/{nsongs}: "
                          f"loss {loss_song/length:.4f}, "
                          f"accuracy {ncorrect_song/length:.4f} "
                          f"({evalset[k][0]})   \n")
            logfile.flush()
            # update total statistics
            ncorrect_tot += ncorrect_song
            loss_tot += loss_song
        # evaluation finished
        length = sum(len(toks) for _, _, toks in evalset)
        logfile.write(f"Completed evaluation: "
                      f"loss {loss_tot/length:.4f}, "
                      f"accuracy {ncorrect_tot/length:.4f}\n")
    return result
