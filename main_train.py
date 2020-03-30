import os
import util.archive
import model.autoencoder
import model.encoder
import model.train
import preprocessing.token
import util.load


trainset, evalset = util.archive.load("ldp5_sorted.zip", 0.8)

autoencoder = model.autoencoder.Autoencoder_SE()
model.train.train(
    autoencoder,
    7,
    trainset,
    evalset)
dir = os.path.join("demo", "model")
autoencoder.save(dir)
with open(os.path.join(dir, "train.txt"), "w") as t, open(os.path.join(dir, "eval.txt"), "w") as e:
    for name, _, _ in trainset:
        t.write(name+"\n")
    for name, _, _ in evalset:
        e.write(name+"\n")



