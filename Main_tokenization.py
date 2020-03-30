import os
import model.eval, util.save
import preprocessing.token
import util.load
import model.autoencoder
import model.encode
import util.archive
import model.train
import util.readlog


#Import MIDI-files and transforms them into token for train purposes

#--------------------------------------------------------------------


def make(source_dir: str, archive_name: str, suffix:str) -> None:
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
   
        if not file.path.endswith(suffix):
            return
        print("Processing file", file.path)
        basename = file.name[:-4]
        multi = pypianoroll.load(file.path)  # Load .npz file
        try:
           melody = preprocessing.melody.get(multi)  # Try to find melody
        except Exception as e:
            if e.args != ("No melody found",):
                raise
            print("> Could not find melody. Skipping.")
            melody_count+=1
            os.remove(file.path)
            return

        try:
            _, dur, toks = preprocessing.token.get(file.path)  # Tokenize
    
        except Exception as e:
            if e.args != ("Variable tempo",):
                raise
            print(" > Tempo must be constant. Skipping.")
            tempo_count+=1
            os.remove(file.path)
            return
        with open(os.path.join(name, basename+".txt"), "w") as file:
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
    print("Number of melody errors: ",melody_count)
    print("Number of tempo errors: ", tempo_count)
    

    return None

#Creates tokenized files and saves them into a zip folder
melody_count=0;
tempo_count=0;
dir = os.path.join("demo", "pretrained")

make(os.path.join(dir, "ldp5_sorted"), "ldp5_sorted",".npz")
