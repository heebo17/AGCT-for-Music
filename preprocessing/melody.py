from typing import List
import pypianoroll


def get(multi: pypianoroll.Multitrack) -> List[int]:
    """
    Returns a list of intergers, containing the indices of multi.tracks
    that contribute to the melody

    TODO:
     - If there is only one track then that one must be the melody. Even if it
        is drums ore bass (or something).
    """
   
    emptytracks = multi.get_empty_tracks()
    melody = []
  
    for (i, tr) in enumerate(multi.tracks):
        if tr.program > 48:
            continue 
        if tr.is_drum:
            continue
        if i in emptytracks:
            continue
        melody.append(i)
    if len(melody) == 0:
        raise Exception("No melody found")

    return melody
