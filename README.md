# Audio Domain Pitch Detector 0
This is an attempt to create a self supervised neural net that learns to do pitch detection. The concept is the same as the SPICE pitch detector by Google AI (https://ai.googleblog.com/2019/11/spice-self-supervised-pitch-estimation.html) whereby you train a network to recognize the relative pitch between two pitch shifted examples. The difference from spice is that this network is intended to be run directly in the audio domain, instead of in the CQT domain used in SPICE. Other than that main difference, most of the concept remains the same as in SPICE.


## Status
This project was put on hold due to taking too much time, and it turns out SPICE will cover my current NN pitch detector needs for the time being. I suspect in principle it could work, perhaps by changing from a CNN to an RNN based architecture, or even Transformer/Reformer architecture, as well as a number of other things. I may reopen this in the future.

## Prerequisites
- sbsms cli
- sox cli
- python3
    - numpy
    - torch
    - soundfile

## instructions for installing sbsms
1. get the current version of sbsms-app http://sbsms.sourceforge.net/ https://sourceforge.net/projects/sbsms/
2. extract somewhere
3. `cd ./sbsms-app-x.x.x/`
4. `./configure LIBS=-lpthread` (may need to include other libraries based on error output)
5. `sudo su`
6. `make`
7. `make install`

If you want to install globally, then you can specify the following during configure
`./configure LIBS=-lpthread --prefix=/absolute/path/to/install/program/at`

Once sbsms is installed you can run it as follows
`sbsms infile outfile rate-start rate-end halfsteps-start halfsteps-end`

For example:
`sbsms blob.wav blobOut.wav .5 .5 0 2 `
will slow down blob.wav by a factor of 2, while simultaneously sliding the pitch up two half-steps, and put the output in blobOut.wav

Original code by Clayton Otey (otey@users.sourceforge.net)


## How this works

The program pitch_detector.py runs a training process. Data is assumed to be stored in a ./data directory. Any wav files found in the directory will be used as data. Audio is pitch shifted using the sbsms cli program, and all pitch shifted audio is cached in a ./cache folder.

Warning: this program has bugs, and also has been mangled from experimentation. The current configuration performs more poorly than ealier/simpler configurations (i.e. without a timing signal, and without the kernel_size=1 layers of the network, and with absolute pitch examples from synthetically generated waveforms, as well as other things I'm probably forgetting)