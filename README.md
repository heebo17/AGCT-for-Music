# Music processing for [AGCT](https://github.com/agct2019/agct)

This is **not a complete program**. It is a collection of Python files that (might) work together to allow some basic training and evaluation of the model described in the concept. There is nothing here to "run".

See the [concept](doc/concept.pdf) for an overview. For a small demo, study the [Jupyter notebook](demo/colab_nb.ipynb) and run it on Google Colaboratory.

## Requirements
* PyTorch (>= 1.2.0)
* pypianoroll (You don't need this if you don't work with midi files **and** fix the global import in `preprocessing.token`, ie. remove all function annotations.)

## Edited by Oliver Heeb
Before you start, some additional data are necessary:
1) Donwload LDP-5-cleansed dataset from https://salu133445.github.io/lakh-pianoroll-dataset/dataset and add it in the folder demo/pretrained in a folder named ldp5.
2) Unzip the file prep/annotation/msd_amglabels_all.csv


##Explanation and Overview
In general, there are four main files. Each with a different Task.
*The main_pred_dataset.py file prepares the dataset for training and transforming into AGCT code. The first part compares the track name of the LDP5 dataset with the genre annotation file “msd_amglabels_all.csv”. Results in creating a new file with a list of all tracks and their labels. The complete list is saved as prep\annotation\”track_names_all.txt”.
The second part selects from the complete dataset (LDP5) the desired number of songs and the desired genre. Then it transfers the songs into a folder called data for further processing by the autoencoder. In the command the number describes how many songs from each file it should transfer. And in the file prep/annotation/genre_name.txt you can write all the desired genres.
*The main_tokenization.py file transforms the mid or npz file from a desired folder into tokenized text files. The tokenized text files then can be used for training the model
The main_train.py file trains the model.
In main_AGCT.py almost all comes together. The model weights are loaded. Secondly, it loads the mid or npz file from the folder “data”. Evaluate the results of the encoding and saving the code in a form for AGCT. In addition, a file as music ontology is created. It is called “ontology.txt” and saved in the folder prep/annotation.
For the moment, I programmed that only one chunk of each song is transformed for AGCT, this you can change in the util/save.py file under the command code at the end. The original part is in comment.
In Eval_AGCT.py is for the moment a draft version of the evaluation. The file reads in all cluster exported from AGCT and transformed it into a table in the text file called results.txt. Showing which song is in which cluster. 

##Process
1) First, run Main_prep_dataset.py
2) Run Main_tokenization, creates a zip folder with all the tokenized songs from ldp5_sorted. Can be used for training the model
3) Run main_train
4) Run main_AGCT.py
5) Take from "demo/pretrained/" the file "agct_dataset.txt" as input for AGCT and from "prep/annotation" the file "ontology.txt" as gene priori list.

Since all these Steps for the ldp5 dataset would take a huge amount of time. There is already a trained model. In addition, you can choose, how many songs it should prepare in main_prep_dataset, by choosing a int in the command "prep.search.copy"

## TODOs and wish list
 * Use the entire dataset for training. It is difficult to train a large dataset, because Google Colaboratory decides to randomly disconnect the runtime. For now, we only used the first 1000 songs, but there are 9944 songs in `all.zip`. (Actually, I think it should be more. I can not believe that only half of the 21425 songs in [lpd-5-cleansed](https://salu133445.github.io/lakh-pianoroll-dataset/dataset#lpd-5) made it through the selection process. I think it should be around 80%. Something must have gone wrong when I tried to create the archive.)
 * Decrease the number of samples per song. For now one "sequence" amounts to 10 tokens, which is about 5 notes. How much information can there be in this short time span? However, increasing the sequence length will increase the time required for training even further. It might also be a good idea to try to restrict training data (eg. sample must start at the beginning of beat) and/or consider variable sized sequences (eg. 2-4 beats). Alternatively, we could just concatenate the samples from each song for faster processing in AGCT.
 * Revise the way we select the melody track. For the training set, we simply selected the piano and strings track. This does not produce good results for all songs and will certainly not work in general. But I suppose you could write entire papers about this problem. (Maybe [this](https://ieeexplore.ieee.org/abstract/document/1565863) is a good start?)
 * Maybe try different token representations (eg. Fig. 7 in [this paper](https://arxiv.org/pdf/1809.04281.pdf)) or add more embeddings (eg. as in [MuseNet](https://openai.com/blog/musenet/#embeddings))




 