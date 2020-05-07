import numpy as np
import simpleaudio as sa
# import matplotlib.pyplot as plt
import music21
import time
import pdb



square   = np.vectorize(lambda x: 1.0 if x < np.pi else -1.0)
square2   = np.vectorize(lambda x: 1.0 if x < np.pi/2 else -1.0)
saw      = np.vectorize(lambda x: x/np.pi - (0 if x < np.pi else 2))
triangle = np.vectorize(lambda x: (2*x/np.pi - (0 if x < np.pi/2 else 4)) if x < np.pi/2 or x > 3*np.pi/2 else 2-2*x/np.pi)
noise    = np.vectorize(lambda x: np.clip(np.random.normal(loc=0.0, scale=0.1), -1.0, 1.0))

waves = ['sine', 'square', 'square2', 'saw', 'triangle', 'noise']

wave_funcs = {
    'sine'      : np.sin,  
    'square'    : square, 
    'saw'       : saw, 
    'triangle'  : triangle,
    'square2'   : square2,
}

def generate_wave(form, amplitude, pitch, duration, FS):
    """
    Generate a waveform with the desired properties and sample rate
    (REMOVED THIS)Note that the duration will be slightly different than specified
    (REMOVED THIS)This is to ensure the waveform starts and ends at a rising zero crossing
    
    Inputs:
      form - string representing type of waveform to create.
        options are sine, square, saw, triangle, cycloid, and pulse
    """
    assert(form in waves)

    if form is not 'noise':

        # num_cycles = int(pitch * duration)
        # cycle_samples = int(FS / pitch)
        # samples = np.linspace(0, 2*np.pi*num_cycles, cycle_samples*num_cycles, endpoint=False) % (2 * np.pi)
        num_samples = int(duration * FS)
        samples = np.linspace(0, 2*np.pi*pitch*duration, num_samples, endpoint=False) % (2 * np.pi)


        return amplitude * wave_funcs[form](samples)

    else:
        return amplitude * np.random.normal(0.0, 0.1, size=int(duration*FS))

    # play(wave, FS, wait=False)
    # plt.plot(wave[0:100])
    # plt.show()


def get_bitwave(wave):
    """convert the wave to 16-bit integer format"""
    return (wave * 32767).astype(np.int16)


def play(wave, FS=16000, wait=True):
    """play the waveform using simpleaudio"""
    bitwave = get_bitwave(wave)
    player = sa.play_buffer(audio_data=bitwave, num_channels=1, bytes_per_sample=2, sample_rate=FS)
    if wait: player.wait_done()


def pitch(name):
    """return the frequency in Hz based on the note name"""
    return music21.pitch.Pitch(name).frequency


def generate_random_sequence(form, duration, FS):
    """generate a random wave sequence of notes"""
    samples = 0 #keep track of the current length generated
    total_samples = int(FS * duration)
    speed = duration / np.random.uniform(low=0.5, high=3*duration)


    #wave contains the wave generated, conditions contains the condition data that matches to it
    #condition label is as follows: [frequency, pitch, is_sine, is_square, is_saw, is_triangle]
    wave = np.zeros(total_samples)
    conditions = np.zeros((2+len(waves), total_samples))
    conditions[2+waves.index(form),:] = 1  #set the one-hot for corresponding wavetype to true

    while samples < total_samples:
        frequency = np.random.uniform(low=pitch('C4'), high=pitch('C5')) # max range is A0-C8
        #length = duration
        length    = np.random.uniform(low=0.1, high=speed)
        amplitude = np.random.uniform(low=0.0, high=0.5)
        
        wave_excerpt = generate_wave(form=form, amplitude=amplitude, pitch=frequency, duration=length, FS=FS)
        new_samples = min(wave_excerpt.shape[0], int(total_samples-samples))
        wave[samples:samples+new_samples] = wave_excerpt[0:new_samples]
        conditions[0, samples:samples+new_samples] = frequency
        conditions[1, samples:samples+new_samples] = amplitude #amplitude is calculated as Mean Absolute Value? 
        samples += len(wave_excerpt)

    return wave, conditions


if __name__ == '__main__':
    FS=16000
    duration = 3
    form = 'sine'
    sample = generate_wave(form=form, amplitude=0.8, pitch=440, duration=duration, FS=FS)
    play(sample, FS)
    # for i in range(10):
    #     form = waves[np.random.randint(low=0, high=len(waves))]
    #     label = generate_random_sequence(form=form, duration=duration, FS=FS)
    #     play(label[0], FS)
