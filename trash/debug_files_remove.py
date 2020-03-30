import os


def process_file(file: os.DirEntry):
    if file.path.endswith(".npz"):
        return
    print("Deleting ", file.path)
    os.remove(file.path)
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
