from typing import List, Tuple
import os
import torch


import preprocessing.token


def toks(dirname: str) -> List[Tuple[str, float, List[Tuple[int, int]]]]:
    """
    Recursively loads songs from all .json files from directory dirname.
    """
    songs = []
    with os.scandir(dirname) as it:
        for direntry in it:
            if direntry.is_dir(follow_symlinks=False):
                # recurse
                songs += toks(direntry.path)
            # file
            name, ext = os.path.splitext(direntry.name)
            if ext == ".json":
                with open(direntry.path) as f:
                    d, t = preprocessing.token.load(f)
                songs.append((name, d, t))
    return songs


def midi(dirname: str) -> List[Tuple[str, float, List[Tuple[int, int]]]]:
    """
    Recursively loads songs from all .mid files from directory dirname.
    """
    songs = []
    with os.scandir(dirname) as it:
        for direntry in it:
            if direntry.is_dir(follow_symlinks=False):
                # recurse
                songs += midi(direntry.path)
            # file
            name, ext = os.path.splitext(direntry.name)
            if ext == ".mid":
                songs.append(preprocessing.token.get(direntry.path))
              
    return songs

def npz(dirname: str) -> List[Tuple[str, float, List[Tuple[int, int]]]]:
    """
    Recursively loads songs from all .npz files from directory dirname.
    """

    import pypianoroll
    songs = []
    name_list=[]
    with os.scandir(dirname) as it:
        for direntry in it:
            new_name, old_name=os.path.split(direntry.path)
            if direntry.is_dir(follow_symlinks=False):
                # recurse
                
                songs += npz(direntry.path)
            # file
            name, ext = os.path.splitext(direntry.name)
            print(old_name)
            if ext == ".npz":
                multi=pypianoroll.load(direntry.path)
                songs.append(preprocessing.token.get(multi))
                name_list+=name
             
    return songs

def prefix(dirname: str) -> List[Tuple[str]]:
        
     #Saves the name as prefix for saving as AGCT raw text file
     #not finished, try a different approach in prefix2
    folder_name=[]

    with os.scandir(dirname) as it:
        for direntry in it:
            if direntry.is_dir():
               new_name, old_name=os.path.split(direntry.path)
               folder_name.append(old_name)
    f=open(os.path.join("demo/pretrained","prefix_list.txt"),"w")
     
    for i in range(len(folder_name)-1):
        f.write(folder_name[i])
        f.write("\t")
        f.write("\n")
    f.write(folder_name[len(folder_name)-1])
    f.close()


    return None



