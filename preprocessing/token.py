from typing import List, Tuple
import io
import json
import numpy
import pypianoroll
import pretty_midi


def tok2int(tok: Tuple[int, int]) -> int:
    l, p = tok
    if l >= 128:
        # cap note length at 127
        l = 127
    elif l == 0 and p >= 128:
        # cap timeshifts at 127
        p = 127
    return 128*l + p


def int2tok(val: int) -> Tuple[int, int]:
    l = val // 128
    p = val % 128
    return l, p


def get(roll,tracks: List[int] = None) -> Tuple[str, float, List[Tuple[int, int]]]:
    """
    Create a token representation of `roll'. It considers only the tracks
    specified by `tracks'.

    Arguments:
     - roll: must be the name of MIDI-file (can also be an instance of
             pypianoroll.Multitrack)
     - tracks: the relevant tracks. If this is set, all other tracks are
               ignored. If this variable not set, it defaults to all of the
               tracks.

    Returns a tuple (name, duration, tokens), where
     - name is set to the basename of the file without the extension (or the
            name specified in the given pypianoroll.Multitrack)
     - duration is the length of a tick (in second)
     - tokens is the list of tokens

    TODO: speed this up.
    """

    assert(isinstance(roll, (str, pypianoroll.Multitrack)))
    if isinstance(roll, str):
        import os.path
        name = os.path.splitext(os.path.basename(roll))[0]
        roll = pypianoroll.parse(roll, beat_resolution=24, name=name)
    if tracks is None:
        empty = roll.get_empty_tracks()
        tracks = [i for i in range(len(roll.tracks)) if i not in empty]

    if roll.beat_resolution != 24:
        raise NotImplementedError("Unsupported beat resolution (samples per be"
                                  "at). I can only handle 24 saples per beat.")
    #for t in roll.tempo[1:]:
    #    if(abs(roll.tempo[0]-t) > 1e-6):
    #       raise Exception("Variable tempo")  #constant tempo
    #duration of a single tick
    duration = 60 / (roll.beat_resolution * roll.tempo[0])

    tokens = []
    if len(tracks) == 0:
        return roll.name, duration, tokens

    # sanity check
    shape = roll.tracks[tracks[0]].pianoroll.shape
    for i in range(1, len(tracks)):
        assert(shape == roll.tracks[tracks[i]].pianoroll.shape)  # size match?
    assert(shape[1] == 128)  # 128 pitch values (columns)

    thresh = 1.  # threshold when a note counts as ``on"
    # find tick when first note starts
    prev = -1
    for tick in range(shape[0]):
        for t in tracks:
            # check if a note starts in track t at current time tick
            for pitch in range(shape[1]):
                if roll.tracks[t].pianoroll[tick, pitch] > thresh:
                    # note is ``on''
                    if tick != 0 and prev == -1:
                        # advance time to current tick
                        tokens.append((0, tick))
                    length = 1
                    while (tick+length < shape[0]
                           and roll.tracks[t].pianoroll[tick+length, pitch] > thresh):
                        # compute length of note
                        length = length+1
                    tokens.append((length, pitch))
                    prev = tick
        if prev != -1:
            break
    # process notes starting at later tick
    for tick in range(prev+1, shape[0]):
        for t in tracks:
            # check if a note starts in track t at current time tick
            for pitch in range(shape[1]):
                if (roll.tracks[t].pianoroll[tick, pitch] > thresh
                        and roll.tracks[t].pianoroll[tick-1, pitch] < thresh):
                    # a new note starts now
                    if (prev != tick):
                        # advance time to current tick
                        tokens.append((0, tick-prev))
                        prev = tick
                    length = 1
                    while (tick+length < shape[0]
                           and roll.tracks[t].pianoroll[tick+length, pitch] > thresh):
                        # compute length of note
                        length = length+1
                    tokens.append((length, pitch))
    tokens.append((0, shape[0]-prev))  # insert a timeshift at the end.
    return roll.name, duration, tokens


def create_roll(name: str,
                duration: float,
                tokens: List[Tuple[int, int]]) -> pypianoroll.Multitrack:
    """
    Turns the token representation back into a pypianoroll. This is mainly for
    debuging purposes. See also the save function further below.
    """
    tokens = fix(tokens)  # fix tokens before processing
    # compute number of rows (= total number of ticks)
    l = 0
    for (length, pitch) in tokens:
        if length == 0:
            l += pitch
    pianoroll = numpy.zeros((l, 128), dtype=numpy.uint8)
    tick = 0
    for (len, pitch) in tokens:
        if len == 0:
            # advance time
            tick += pitch
            continue
        # add note
        pianoroll[tick:tick+len, pitch] = 100
    track = pypianoroll.Track(pianoroll)
    return pypianoroll.Multitrack(tracks=[track], tempo=60/(24*duration),
                                  beat_resolution=24, name=name)


def fix(tokens: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Fix the token representation (such that it satisfies some essential
    invariants and can be processed in create_roll).
    For now, it just:
     - orders notes that start simultaneously according to pitch
     - combines successive time shift tokens into one
     - makes sure the time shift tokens add up to the length of the song
    It does NOT:
     - check if two notes intersect
     - combine intersecting notes into a longer one.

    TODO: check if two notes intersect (efficiently) and combine any that do
    """
    result = []
    time = 0
    maxtime = 0
    wait = 0
    for (length, pitch) in tokens:
        if length == 0:
            # time shift token
            wait += pitch  # combine all successive time shifts
            continue
        # now its a note token
        if wait > 0:
            # add timeshift token
            result.append((0, wait))
            time += wait
            wait = 0
        # add note token
        i = len(result)
        while i-1 >= 0 and (result[i-1][0] != 0 and result[i-1][1] > pitch):
            i -= 1
        result.insert(i, (length, pitch))
        maxtime = max(maxtime, time+length)
    # append trailing time shifts, if any. At least advance time until end
    result.append((0, max(wait, maxtime-time)))
    return result


def save(name: str, duration: float, tokens: List[Tuple[int, int]]) -> None:
    """
    Saves the song in a MIDI file called `name'.mid
    """
    roll = create_roll(name, duration, tokens)
    roll.write(name+".mid")
    return None


def dump(file_object: io.TextIOBase,
         duration: float,
         tokens: List[Tuple[int, int]],
         *arguments,
         **keywords) -> None:
    """
    Saves duration and tokens into a file (file object, obtained through open)
    arguments and keywordes are passed to json.dump
    """
    dict = {
        "tick_duration": duration,
        "tokens": tokens,
    }
    json.dump(dict, file_object, *arguments, **keywords)
    return None


def load(file_object: io.TextIOBase) -> Tuple[float, List[List[int]]]:
    """
    Loads duration and tokens from a file (file object, obtained through open)

    (This actually returns the tokens as a 2-sized list, because json does not
     support the tuple data-structure, so it saves tokens as lists.)
    """
    dict = json.load(file_object)
    return dict["tick_duration"], dict["tokens"]
