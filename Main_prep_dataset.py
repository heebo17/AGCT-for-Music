import os

import prep.genre
import prep.search 

#Compare the the Track IDs given with the annotation
#and create a txt file with all the Tracks ID
#and their annotation
#The file is called "prep/annotation/track_names_all.txt


dir=os.path.join("prep","annotation")

#prep.genre.prefix(os.path.join(dir,"id_cleansed.csv")
#                  ,os.path.join(dir,"msd_amglabels_all.csv")
#                  ,os.path.join("demo/pretrained","ldp5"))
#prep.make_ontology.make_ontology(os.path.join(dir,"track_names_all.txt"))

#Search all the files from track_names_all.txt in the ldp5_sorted database
#and saves them in a new folder called data for furthere processing.
#The first int in the command gives the number of song for each genre defined in 
#the genra_name.txt file if the int is equal to zero, all files from ldp5_sorted are processed.


prep.search.copy(20,os.path.join(dir,"track_names_all.txt")
                 ,os.path.join("demo","pretrained")
                 ,os.path.join(dir,"genre_name.txt"))