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