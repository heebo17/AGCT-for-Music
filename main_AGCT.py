import os
import model.eval, util.save
import preprocessing.token
import util.load
import model.autoencoder
import model.encode
import util.archive
import model.train
import util.readlog

#Creating a model

autoencoder = model.autoencoder.Autoencoder_SE()



#Loading saved model

dir = os.path.join("demo", "pretrained")
dir2 = os.path.join("prep","annotation")
autoencoder = model.autoencoder.load(dir)
# also show params.json
with open(os.path.join(dir, "params.json")) as params:
    print(params.read())


#loading MIDI file and saves all tracknames to the prep/annotation/prefix_list.txt
songs=util.load.npz(os.path.join(dir,"data"))
util.load.prefix(os.path.join(dir,"data"))




#Visualize training accuracy

#import util.readlog
#util.readlog.plot(os.path.join(dir, "train.log"))


#Evaluating and Saving

pred = model.eval.eval(autoencoder, songs)
#save songs with -pred attached to their name
util.save.midi(os.path.join(dir, "midi-pred"), [(name+"-pred", dur, toks) for name, dur, toks in pred])


#Writing AGCT-file

#Load prefix as a list:
with open(os.path.join(dir,"prefix_list.txt"),"r") as f:
  
    prefix=[]
    for line in f:
        fields=line.split("\t")
        prefix.append(fields[0])
c = model.encode.encode(autoencoder, songs)
util.save.code(os.path.join(dir, "agct_dataset.txt"), c, prefix)