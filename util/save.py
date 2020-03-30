from typing import List, Tuple
import os
import torch
import random
import preprocessing.token


def toks(dirname: str,
         songs: List[Tuple[str, float, List[Tuple[int, int]]]]):
    """
    Saves all songs in json format in directory dirname.
    """
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    for name, dur, toks in songs:
        fname = os.path.join(dirname, name+".json")
        with open(fname, "w") as f:
            preprocessing.token.dump(f, dur, toks)


def midi(dirname: str,
         songs: List[Tuple[str, float, List[Tuple[int, int]]]]):
    """
    Saves all songs in midi format in directory dirname.
    """
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    for name, dur, toks in songs:
        fname = os.path.join(dirname, name)
        preprocessing.token.save(fname, dur, toks)

        
def npz(dirname: str,
         songs: List[Tuple[str, float, List[Tuple[int, int]]]]):
    """
    Saves all songs in midi format in directory dirname.
    """
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    for name, dur, toks in songs:
        fname = os.path.join(dirname, name)
        preprocessing.token.save(fname, dur, toks)


def code(filename: str,
         code: List[List[torch.tensor]],
         prefix: List[str]):
    """
    Saves each code-list in filename, in a format readable by AGCT.
    The gene-name of each j-th code-vector from the i-th song (the name of the
    code vector code[i][j]) will be {prefix[i]}_{j}.
    Prefixes should not contain any spaces (or any other characters that
    interfer with string tokenization)
    """
    if prefix is None:
        # set prefix to "AA", "AB", "AC", ...
        prelen = 1
        while 26**prelen < len(code):
            prelen += 1
        prefix = [chr(i) for i in range(ord("A"), ord("Z")+1)]
        prelen -= 1
        while prelen > 0:
            prefix = [
                chr(i)+s for i in range(ord("A"), ord("Z")+1) for s in prefix
                ]
            prelen -= 1



    length = sum(len(l) for l in code)
    nsamples = len(code[0][0])
    with open(filename, "w") as f:
        # write header
        f.write("Species\t")
        f.write("ontology.txt\n")
        f.write("E (entity)	1	E1\n")
        f.write("EG (entity groupping)	1	1\n")
        time_str = f"T (time)	{nsamples}"
        for k in range(nsamples):
            time_str += f"	{k}"
        f.write(time_str+"\n")
        f.write("R (replicate)	1\n")
        f.write(f"Data	{len(code)}\n")


        # write samples for all chunks of one song
        #for i in range(len(code)):
        #    for j in range(len(code[i])):
        #        data_str = f"{prefix[i]}_{j}"
        #        for k in range(len(code[i][j])):
        #            val = code[i][j][k].item()
        #            data_str += f"	{val:7f}"
        #        f.write(data_str+"\n")

        
        
        #write one sample of each song
        #for better processing in AGCT
        for i in range(len(code)):
            j=random.randrange(0,len(code[i]),1)
            data_str = f"{prefix[i]}"
            for k in range(len(code[i][j])):
                val = code[i][j][k].item()
                data_str += f"	{val:7f}"
            f.write(data_str+"\n")

