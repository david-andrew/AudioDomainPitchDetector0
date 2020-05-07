import pdb

import os
import sys
from itertools import islice
import subprocess
from subprocess import DEVNULL

import numpy as np
import torch
from torch.utils.data import Dataset
# from math import log as ln, ceil, floor
from math import ceil
import soundfile
# from utilities import noise


#TODO: look into using virtual memory for all of the data processing
# e.g. /dev/shm/my_folder which hosts my_folder on ram

#TODO: look into loading directly the datasets in torchaudio
# probably will need to load audio, clip to 1 second and then pitch shift on the fly, rather than caching on the disk...


def get_dict_key_i(dct, i):
    it = iter(dct)
    next(islice(it, i, i), None)
    return next(it)

# def get_white_noise(num_samples, max_scale=1.0):
#     return noise(np.arange(num_samples)) * max_scale



class ShiftAudioLoader(Dataset):
    """class for loading and pitch shifting monophonic audio in a memoized fashion"""

    def __init__(self, root_path=os.path.abspath('.'), duration=1, AR=16000, max_shift_up=6, max_shift_down=6):

        self.duration = duration                #seconds
        self.AR = AR                            #Audio Rate (samples per second)
        self.window = int(duration * AR)        #number of samples in a clip
        self.max_shift_up = max_shift_up        #semitones
        self.max_shift_down = max_shift_down    #semitones

        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'data')
        self.cache_path = os.path.join(root_path, 'cache')
        if not os.path.isdir(self.cache_path):
            os.mkdir(self.cache_path)

        self.files = {}
        self.update_training_list()


    def __len__(self):
        shift_range = self.max_shift_up + self.max_shift_down + 1
        return len(self.files) * shift_range * shift_range #encode file idx and 2 shift values in a single number

    def __getitem__(self, idx):

        #compute the pitch shift from the index
        shift_range = self.max_shift_up + self.max_shift_down + 1
        shift0 = idx % shift_range - self.max_shift_down
        idx = (idx - idx % shift_range) // shift_range
        shift1 = idx % shift_range - self.max_shift_down
        idx = (idx - idx % shift_range) // shift_range

        #collect the shifted versions of the item (from cache if exists, else aslo make cache entry)
        raw_path = get_dict_key_i(self.files, idx)
        item0 = self.memoized_load_shift_sample(raw_path, shift0)
        item1 = self.memoized_load_shift_sample(raw_path, shift1)

        assert(item0.shape == item1.shape)

        #clip both audio tracks to a fixed duration
        window = int(self.duration * self.AR)
        if item0.shape[0] < window:
            item0 = np.pad(item0, pad_width=((0, window-item0.shape[0]),))
            item1 = np.pad(item1, pad_width=((0, window-item1.shape[0]),))

        #get start and stop indices to clip the audio clips at
        start, stop = self.get_clip_idx(item0)

        #clip the audio, and maybe eventually handle fade in and fade out
        item0 = self.clip_audio(item0, start, stop)
        item1 = self.clip_audio(item1, start, stop)

        #compute the relative pitch between the two samples in semitones
        shift = shift1 - shift0

        #convert everything to torch tensors
        item0 = torch.tensor([item0], dtype=torch.float32) #convert to 2D tensor for a 1-channel audio signal of length (stop-start)
        item1 = torch.tensor([item1], dtype=torch.float32)
        shift = torch.tensor(shift, dtype=torch.float32)

        return shift, item0, item1


    def get_clip_idx(self, sound):
        """return the start and stop index of to clip the audio (TODO, clip so that the audio contains at least some sound)"""
        window = int(self.duration * self.AR)
        
        #verify that audio is at least requested size
        assert(sound.shape[0] >= window)

        #compute a random offset into the audio to clip at
        offset = np.random.randint(sound.shape[0] - window + 1)
        start = offset
        stop = start + window - 1
        return start, stop

    def clip_audio(self, sound, start, stop):
        """return the audio cliped to the specified indices. TODO->make this fade in and fade out (optionally?)"""
        sound = sound[start:stop]

        #if self.fade_samples:
        # multiply by ramp up and ramp down from zero over several milliseconds

        return sound

    def memoized_load_shift_sample(self, raw_path, shift):
        """load the specified sample shifted by the specified amount. Cache so that future loads are fast"""
        # try:
        cache = self.files[raw_path]
        
        if shift not in cache:
            #separate .wav extension
            #replace all periods and slashes with 
            raw_handle, raw_ext = os.path.splitext(raw_path)
            assert(raw_ext == '.wav')

            #create a deterministic and unique filename for the cache. Basically convert the path to a filename
            #replace characters that the cli programs can't handle for the unique filename. i.e. remove directory slashes, and any internal periods
            path_handle = raw_handle.replace('/', '-').replace('\\', '-').replace('.', '-') + raw_ext

            #if the unshifted version hasn't been cached, create it by downsampling the original
            if 0 not in cache:
                base_cache_path = os.path.join(self.cache_path, f'shift_0_{path_handle}')
                if not os.path.exists(base_cache_path):
                    subprocess.Popen(['sox', raw_path, '-r', f'{self.AR}', base_cache_path], stdout=DEVNULL, stderr=DEVNULL).wait()
                    assert(os.path.exists(base_cache_path))
                cache[0] = base_cache_path

            #if the requested shift isn't the 0-shift we just created, create/cache it
            if shift != 0:
                #get the path to the unshifted version
                base_cache_path = cache[0]
                new_cache_path = os.path.join(self.cache_path, f'shift_{shift}_{path_handle}')
                if not os.path.exists(new_cache_path):
                    tmp_path = os.path.join(self.cache_path, f'tmp_{shift}_{path_handle}')
                    subprocess.Popen(['sbsms', base_cache_path, tmp_path, '1', '1', f'{shift}', f'{shift}'], stdout=DEVNULL, stderr=DEVNULL).wait()
                    subprocess.Popen(['sox', tmp_path, '-r', f'{self.AR}', new_cache_path], stdout=DEVNULL, stderr=DEVNULL).wait() #resample the output of sbsms to the desired rate
                    try:
                        assert(os.path.exists(tmp_path))
                        assert(os.path.exists(new_cache_path))
                    except:
                        pdb.set_trace()
                    os.remove(tmp_path)
                cache[shift] = new_cache_path

        cache_path = cache[shift]
        sound, sr = soundfile.read(cache_path)
        assert(sr == self.AR)
        # except:
            # sound = get_white_noise(int(self.duration * self.AR)) #if we failed to laod the file, fallback to white noise

        return sound



    def update_training_list(self):
        """save the paths of every wav file in the data directory"""
        for root, _, files in os.walk(self.data_path):
            for filename in files:
                if not filename.startswith('.') and os.path.splitext(filename)[1] == '.wav':  #only mark non-hidden wav files
                    filepath = os.path.join(root, filename)
                    self.files[filepath] = {}



if __name__ == '__main__':
    
    data = ShiftAudioLoader()
    from utilities import play
    for i in range(len(data)):
        idx = i#np.random.randint(len(data))
        print(f'idx: {idx}/{len(data)}...', end='')
        sys.stdout.flush()
        shift, x0, x1 = data[idx]
        print(f'shift: {shift}...', end='')
        sys.stdout.flush()
        # play(x0, data.AR)
        # play(x1, data.AR)
        print('done')

