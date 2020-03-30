import os
import matplotlib.pyplot

import preprocessing.token

note = []
shift = []


def process_file(file: os.DirEntry):
    print("Processing file", file.path)
    with open(file.path) as f:
        dur, toks = preprocessing.token.load(f)
        for tok in toks:
            if tok[0] != 0:
                note.append(tok[0])
            else:
                shift.append(tok[1])
    return


dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir)
for entry in os.scandir("json"):
    if entry.is_file() and (entry.name.endswith(".txt") or entry.name.endswith(".json")):
        process_file(entry)
    else:
        print("Strange dir entry ", entry.path, flush=true)
        os.abort()

bins = [4*i for i in range(80)]
matplotlib.pyplot.hist(note, bins=bins, density=True, facecolor='black')
matplotlib.pyplot.hist(shift, bins=bins, density=True, facecolor='grey')
matplotlib.pyplot.title("Length distribution")
matplotlib.pyplot.legend(labels=("Notes", "Timeshifts"))
matplotlib.pyplot.savefig("hist")
matplotlib.pyplot.clf()
