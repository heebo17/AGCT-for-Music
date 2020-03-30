import os
import numpy as np
import pypianoroll
import preprocessing.token
import preprocessing.melody
import matplotlib.pyplot


def process_file(file: os.DirEntry):
    if not file.path.endswith(".npz"):
        return
    print("Processing file", file.path)
    basename = file.path[:-4]
    multi = pypianoroll.load(file.path)
    try:
        melody = preprocessing.melody.get(multi)
    except Exception as e:
        if e.args != ("No melody found",):
            raise
        print(" > Could not find melody. Skipping.")
        return
    try:
        _, dur, toks = preprocessing.token.get(multi, melody)
    except Exception as e:
        if e.args != ("Variable tempo",):
            raise
        print(" > Tempo must be constant. Skipping.")
        return

    # Try to save and load tokens
    with open(basename+".txt", "w+") as f:
        preprocessing.token.dump(f, dur, toks, indent=2)
        f.seek(0, 0)
        dur2, toks2 = preprocessing.token.load(f)
    assert(dur == dur2)
    assert(len(toks) == len(toks2))
    for i in range(len(toks)):
        assert(toks[i][0] == toks2[i][0] and toks[i][1] == toks2[i][1])

    # Try to convert to integers and back again
    ints = [preprocessing.token.tok2int(t) for t in toks2]
    tok3 = [preprocessing.token.int2tok(i) for i in ints]
    assert(len(toks) == len(tok3))
    for i in range(len(toks)):
        if toks[i][0] != 0:
            # A regular note
            assert(min(toks[i][0], 127) == tok3[i][0] and toks[i][1] == tok3[i][1])
        else:
            # a timeshift
            assert(tok3[i][0] == toks[i][0] == 0 and min(toks[i][1], 127) == tok3[i][1])
    roll = preprocessing.token.create_roll(basename, dur2, tok3)
    pypianoroll.write(multi, basename+".orig.mid")
    pypianoroll.write(roll, basename+".melody.mid")
    notes = []
    waits = []
    for tok in toks:
        if tok[0] != 0:
            notes.append(tok[0])
        else:
            waits.append(tok[1])
    matplotlib.pyplot.hist(notes, facecolor='black')
    matplotlib.pyplot.hist(waits, facecolor='grey')
    matplotlib.pyplot.title("Histogram of length distribution")
    matplotlib.pyplot.legend(labels=("Notes", "Timeshifts"))
    matplotlib.pyplot.savefig(basename)
    matplotlib.pyplot.clf()
    return


def process_dir(path: str):
    for entry in os.scandir(path):
        if entry.is_dir():
            process_dir(entry.path)
        elif entry.is_file():
            process_file(entry)
        else:
            print("Strange dir entry ", entry.path, flush=true)
            os.abort()

dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir)
process_dir("Lakh Pianoroll Dataset - 5_cleansed")
