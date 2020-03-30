from typing import List, Tuple
import os
import torch
import shutil


def prefix(filename_trackID: str, filename_genre: str,dirname: str):
    dir=os.path.join("prep","annotation")
    #Replaces the filename with the genre and set it as prefix for the AGCT file
    f=open(os.path.join(dir,"track_names.txt"),"w")
    #1) load file with all the track ID
    dir2 = os.path.join("demo", "pretrained")
    os.mkdir("demo\\pretrained\\ldp5_sorted")
    with open(filename_trackID,'r') as f_ID:
        i=0
        for line in f_ID:
            fields2=line.split('\n')
            track_name=fields2[0]
            print(track_name)
            with open(filename_genre,'r') as f_genre:
                for line in f_genre:
                    fields=line.split(';')
                    tr_name_genre=fields[0]
                    genre=fields[1]
                    styles=fields[2]
                    moods=fields[3]
                    themes=fields[4]
                    if track_name==tr_name_genre:

                        search(track_name,os.path.join(dir2,dirname)
                               ,os.path.join(dir2,"ldp5_sorted",track_name))

                        f.write(tr_name_genre)
                        f.write(";")
                        f.write(genre)
                        f.write(";")
                        f.write(styles)
                        f.write(";")
                        f.write(moods)
                        f.write(";")
                        f.write(themes)
                        #f.write("\n")
                        i+=1
                        print(i)
                        break
    f.close()            
    return None

def search(trackname: str, path: str, savepath: str):

    for root, dirs, files in os.walk(path):
        
        word=list(trackname)
        for i in range(2,5):
            if word[i] in dirs:
                path=os.path.join(path,word[i])
                i+=1
            else:
                exit()
        folder=os.path.join(path,trackname)
        shutil.copytree(folder, savepath, symlinks=False, ignore=None)
        path=os.path.join(savepath)
        basename=os.listdir(path)
        if basename==[]:
            break
        basename=basename[0]
        name, ext=os.path.splitext(basename)
        trackname+=ext
        os.rename(os.path.join(path,basename),os.path.join(path,trackname))
        break
    return None




