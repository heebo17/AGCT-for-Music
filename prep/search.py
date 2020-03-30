import os
import shutil
import string

def copy(count: int, filename: str, dirname: str, filename2: str):
    dir = os.path.join("demo", "pretrained")
    dir2=os.path.join("prep","annotation")
    shutil.rmtree("demo\\pretrained\\data")
    os.mkdir("demo\\pretrained\\data")

    if count==0:
        with open(filename, "r") as f:
            for line in f:
                fields=line.split(";")
                track_name=fields[0]
                search(track_name,os.path.join(dir,"lpd5")
                               , os.path.join(dir,"data",track_name))
                       
                
    else:   
        with open (filename2) as f2:
            for line in f2:
                genre_name=line.split(";")
                num=len(genre_name)
        with open(filename, "r") as f:

            count2=[0]*num
            for line in f:
                fields=line.split(";")
                track_name=fields[0]
                if fields[0]=="\n":
                    return None
                genre=fields[1]

                j=0
                for j in range(0,num):
                    if genre==genre_name[j]:
                        if count2[j]<count:
                            search(track_name,os.path.join(dir,"lpd5")
                               , os.path.join(dir,"data",track_name))
                            count2[j]+=1
                            break
 
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
        if not basename:
            print("1")
            return
        basename=basename[0]
        name, ext=os.path.splitext(basename)
        trackname+=ext
        os.rename(os.path.join(path,basename),os.path.join(path,trackname))
        break
    return None



