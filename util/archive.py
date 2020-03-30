from typing import List, Tuple

import os
import preprocessing.token


def load(name: str, ratio: float=0.75) -> Tuple[List[
        Tuple[str, float, List[Tuple[int, int]]]]]:
    """
    Reads songs from an archive into two lists including their names.
    arguments:
        - name: archive containing json files of the songs
            (as created with make)
        - ratio: the ratio when splitting up the lists
    This function return two lists, the first one containing ratio-fraction of
    all the songs. An entry (called song) in each list is a tuple containing:
        - the name (in the archive)
        - the tick duration
        - the list of tokens
    """
    import zipfile
    with zipfile.ZipFile(name) as ar:
        all = []
        for member in ar.infolist():
            if not member.is_dir():
                # archive member must be json file
                assert(member.filename.endswith(".txt")
                       or member.filename.endswith(".json"))
                with ar.open(member.filename) as f:
                    # load
                    name = os.path.splitext(
                        os.path.basename(member.filename)
                        )[0]
                    dur, toks = preprocessing.token.load(f)
                    all.append((name, dur, toks))
        thresh = int(len(all)*ratio)
    return all[:thresh], all[thresh:]  # split


def make(source_dir: str, archive_name: str) -> None:
    """
        Recursively processes the directory source_dir and computes the token
        representation of each pianoroll (.npz file) it encounters.
        It saves all token representations to the specified archive name.

        THIS WILL TAKE A LONG TIME! Use the provided archive instead of
        of creating your own.

        TODO: Build archive incrementally instead of creating tmp dir.
    """
    import pypianoroll
    import shutil
    import preprocessing.melody

    name = archive_name.rstrip(".zip")
    os.mkdir(name)  # create temporary directory to save tokens

    def process_file(file: os.DirEntry):
        if not file.path.endswith(".npz"):
            return
        print("Processing file", file.path)
        basename = file.name[:-4]
        multi = pypianoroll.load(file.path)  # Load .npz file
        try:
            melody = preprocessing.melody.get(multi)  # Try to find melody
        except Exception as e:
            if e.args != ("No melody found",):
                raise
            print(" > Could not find melody. Skipping.")
            return
        try:
            _, dur, toks = preprocessing.token.get(multi, melody)  # Tokenize
        except Exception as e:
            if e.args != ("Variable tempo",):
                raise
            print(" > Tempo must be constant. Skipping.")
            return
        with open(os.path.join(name, basename+".json"), "w") as file:
            preprocessing.token.dump(file, dur, toks)  # Save in temporary dir
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

    process_dir(source_dir)
    shutil.make_archive(name, 'zip', name)  # make zip archive
    shutil.rmtree(name)  # remove temporary directory

    return None
