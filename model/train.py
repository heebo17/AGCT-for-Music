from typing import List, Tuple, Generator
import sys
import io
import datetime
import random
import torch

import model.autoencoder
import util.data
import model.eval

seq_len = 10  # length of a chunk of a sequence
batch_size = 64  # size of a training mini-batch
def tf_gen():
    # decrease to 0.1 within 100k
    for i in range(100000):
        thresh = .7 + (i/100000.)*(.1-.7)
        yield [random.random() < thresh for _ in range(1, seq_len)]
    while True:
        yield [random.random() < 0.1 for _ in range(1, seq_len)]
# learning parameters
learning_rate = 0.001  # should be around 0.001
momentum = 0.0  # MBZ
def lr_dec_fac(epoch):
    # decrease lr by a factor of 0.6 every epoch
    return 0.6**epoch
# other
print_every = 1000  # summarize some stats after print_every batches

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train(autoencoder: torch.nn.Module,
          epochs: int,
          trainset: List[Tuple[str, float, List[Tuple[int, int]]]],
          evalset: List[Tuple[str, float, List[Tuple[int, int]]]]=None,
          maxsec: int=None,
          logfile: io.TextIOBase=sys.stdout):
    """
    Train autoencoder on trainset for epochs iterations, but at most for
    approximately maxtime seconds.
    autoencoder: an instance of a class defined in model.autoencoder
    epochs: number of iterations through the training set
    trainset: list of songs
    evalset: list of songs (can be None)
    maxsec: train at most for maxsec seconds (can be None)
    logfile: where to write some statistics (default: stdout)

    TODO:
     - come up with an approximate ETA
    """
    autoencoder.to(device)
    logfile.write("Training on "+str(device)+"\n")

    dataset = util.data.dataset(trainset, seq_len, device)
    sampler = util.data.sampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=util.data.collate_fn,
        drop_last=True)
    nsongs = len(trainset)
    nbatchs = len(loader)
    logfile.write(f"Doing {epochs} iterations on {nsongs} songs. Each epoch "
                  f"consists of {nbatchs} mini-batches of size {batch_size}."
                  f"\n")

    if maxsec is not None:
        stoptime = datetime.datetime.now() + datetime.timedelta(seconds=maxsec)
        logfile.write("Training no longer than "+stoptime.ctime()+".\n")
    else:
        stoptime = None

    optim = torch.optim.SGD(autoencoder.parameters(), lr=learning_rate,
                            momentum=momentum)
    loss_fn = torch.nn.NLLLoss(reduction="sum")  # Compute average manually
    schduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_dec_fac)
    teacher = tf_gen()
    # TODO
    # logfile.write(f"{nabtchs*42}h expected runtime")
    ntrain = 0
    for e in range(epochs):
        logfile.write(f"Epoch {e}/{epochs}\n")
        try:
            ntrain += _train_epoch(
                autoencoder,
                optim,
                loss_fn,
                loader,
                teacher,
                stoptime,
                logfile)
        except RuntimeError as e:
            if len(e.args) != 2 or e.args[0] != "Timelimit reached":
                raise
            ntrain += e.args[1]
            logfile.write(f"Timelimit reached. Did {ntrain} trainings.\n")
            return
        if evalset is not None:
            _ = eval.eval(autoencoder, evalset, logfile)
        schduler.step()  # adjust learning rate
    logfile.write(f"Training completed. Did {ntrain} trainings.\n")
    return None


def _train_epoch(autoencoder: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 loader: torch.utils.data.DataLoader,
                 teacher: Generator[List[bool], None, None],
                 stoptime: datetime.datetime,
                 logfile: io.TextIOBase):
    """
    Train for one epoch.
    """
    autoencoder.train()
    nbatchs = len(loader)
    # Stats
    tot = [
        0,  # 0: number of predctions without tf
        0,  # 1: number of predictions with tf
        0,  # 2: number of correct predictions without tf
        0,  # 3: number of correct predictions with tf
        0.  # 4: loss
    ]
    part = [0, 0, 0, 0, 0, 0.]
    for b, t in enumerate(loader):
        if stoptime is not None and stoptime < datetime.datetime.now():
            # haven't trained current iteration
            raise RuntimeError("Timelimit reached", b)
        if b % print_every == 0 and b != 0:
            for i in range(len(tot)):
                tot[i] += part[i]
            # print summary
            # Assume print_every is large enough so that we get at least
            # one tf and one non-tf prediction. Otherwise we get 0./0. which
            # is unfortunately not NaN in Python but throws ZeroDivisionError.
            npredict = print_every*batch_size*seq_len
            assert(part[0]+part[1] == print_every*batch_size*seq_len)
            loss_avg = part[4] / npredict
            acc = (part[2]+part[3]) / npredict
            tf_ratio = part[1] / npredict
            logfile.write(f" Batch {b-print_every}-{b}: "
                          f"loss {loss_avg:.4f}, accuracy {acc:.4f} "
                          f"({part[2]/part[0]:.4f}, {part[3]/part[1]:.4f}), "
                          f"TF ratio {tf_ratio:.2f}\n")
            logfile.flush()
            part = [0, 0, 0, 0, 0, 0.]
        # train the current batch
        optim.zero_grad()
        tf = next(teacher)
        p, pred = autoencoder(t, tf)
        loss = loss_fn(
            p.view(seq_len*batch_size, autoencoder.dict_size),
            t.view(seq_len*batch_size)
            )
        loss.backward()
        optim.step()
        # collect statisticts of current batch
        npredict = 0
        npredict_tf = 0
        ncorrect = 0
        ncorrect_tf = 0
        # first predictions in batch never teacher-forced
        npredict += batch_size
        for j in range(batch_size):
            if t[0, j] == pred[0, j]:
                ncorrect += 1
        # later predictions in batch might be teacher forced
        for i in range(1, seq_len):
            if tf[i-1]:
                # row i of current batch was predicted with teacher forcing
                npredict_tf += batch_size
                for j in range(batch_size):
                    if t[i, j] == pred[i, j]:
                        ncorrect_tf += 1
            else:
                # row i of current batch was predicted without teacher forcing
                npredict += batch_size
                for j in range(batch_size):
                    if t[i, j] == pred[i, j]:
                        ncorrect += 1
        assert(batch_size*seq_len == npredict+npredict_tf)
        # # print some live statistics
        # acc = (ncorrect+ncorrect_tf) / (batch_size*seq_len)
        # loss_avg = loss.item() / (batch_size*seq_len)
        # logfile.write(f"  Batch {b}/{nbatchs}: "
        #               f"loss {loss_avg:.4f}, accuracy {acc:.4f}, "
        #               f"TF ratio {npredict_tf/(batch_size*seq_len):.2f}   \r")
        # logfile.flush()
        part[0] += npredict
        part[1] += npredict_tf
        part[2] += ncorrect
        part[3] += ncorrect_tf
        part[4] += loss.item()
    # epoch finished (ie. all songs in archive processed once)
    # add remaining stats and print
    b += 1
    for i in range(len(tot)):
        tot[i] += part[i]
    npredict = b*batch_size*seq_len
    assert(npredict == tot[0]+tot[1])
    loss_avg = tot[4] / npredict
    acc = (tot[2]+tot[3]) / npredict
    logfile.write(f"Epoch finished ({b} trainings), "
                  f"loss {loss_avg:.4f}, accuracy {acc:.4f} "
                  f"({tot[2]/tot[0]:.4f}, {tot[3]/tot[1]:.4f})\n")
    return b
