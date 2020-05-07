import pdb

from math import ceil, floor, pi
import torch
import torch.nn as nn
import torch.nn.functional
from utilities import waves, generate_wave


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return nn.functional.gelu(x)

def add_conv_lrelu(in_channels, out_channels, kernel_size=3, stride=1, activation=True):
    padding = floor(kernel_size/2)

    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(0.1) if activation else nn.Sequential()
    )

def add_conv_gelu(in_channels, out_channels, kernel_size=3, stride=1, activation=True):
    padding = floor(kernel_size/2)

    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        GELU() if activation else nn.Sequential()
    )

def net_pitch(Hz):
    """convert Hz to network pitch in the range of (0, 1). Expects torch.tensor as input"""
    return torch.log(Hz) / 10

def net_pitch_inv(a):
    """convert a network pitch back to Hz"""
    return torch.exp(10 * a)

def net_semitone_size(semitones=torch.tensor(1.0)):
    """return the space between the specified number of semitones using the network pitch scale"""
    # return (net_pitch(torch.tensor(440.0)) - net_pitch(torch.tensor(220.0))) / 12 * semitones
    return torch.log(torch.tensor(2 ** (1/120))) * semitones

def net_semitone_size_inv(net_pitch_delta):
    return net_pitch_delta / net_semitone_size()


class AutoPitcherNet(nn.Module):
    def __init__(self, channels=100, add_layer=add_conv_gelu, position_dims=10):
        super(AutoPitcherNet, self).__init__()

        assert(position_dims % 2 == 0) 
        self.position_dims = position_dims #number of dimensions for positional encoding

        self.beginning = add_layer(1 + position_dims, channels, kernel_size=1)
        self.p0 = add_layer(channels, channels, kernel_size=1)
        self.p1 = add_layer(channels, channels, kernel_size=1)
        self.p2 = add_layer(channels, channels, kernel_size=1)
        self.p3 = add_layer(channels, channels, kernel_size=1)
        self.p4 = add_layer(channels, channels, kernel_size=1)
        self.p5 = add_layer(channels, channels, kernel_size=1)
        self.p6 = add_layer(channels, channels, kernel_size=1)
        self.p7 = add_layer(channels, channels, kernel_size=1)
        self.p8 = add_layer(channels, channels, kernel_size=1)


        #should be looking for zero crossings and those types of features
        # self.layer0 = add_layer(1 + position_dims, channels)
        self.layer0 = add_layer(channels, channels)
        self.layer1 = add_layer(channels, channels)
        self.layer2 = add_layer(channels, channels)

        #convolutional pooling
        self.layer3 = add_layer(channels, channels, stride=2)
        self.layer4 = add_layer(channels, channels, stride=2)
        self.layer5 = add_layer(channels, channels, stride=2)
        self.layer6 = add_layer(channels, channels, stride=2)
        self.layer7 = add_layer(channels, channels, stride=2)
        self.layer8 = add_layer(channels, channels, stride=2)
        self.layer9 = add_layer(channels, channels, stride=2)

        #final layers for pitch determination
        self.layer10 = add_layer(channels, channels)
        self.layer11 = add_layer(channels, channels)
        # self.layer12 = add_layer(channels, 1, activation=False) #skip the final activation on the final layer
        self.layer12 = add_layer(channels, channels)

        self.f0 = add_layer(channels, channels, kernel_size=1)
        self.f1 = add_layer(channels, channels, kernel_size=1)
        self.f2 = add_layer(channels, channels, kernel_size=1)
        self.f3 = add_layer(channels, channels, kernel_size=1)
        self.f4 = add_layer(channels, channels, kernel_size=1)
        self.f5 = add_layer(channels, channels, kernel_size=1)
        self.f6 = add_layer(channels, channels, kernel_size=1)
        self.f7 = add_layer(channels, channels, kernel_size=1)
        self.f8 = add_layer(channels, channels, kernel_size=1)
        self.final = add_layer(channels, 1, kernel_size=1, activation=False)

        # self.dropout = nn.dropout(0.5) #dont need with GELU

    def forward(self, x):

        #activation
        a = x

        #add a positional encoding to the input signal
        zeros = torch.zeros((a.shape[0], self.position_dims, a.shape[2]), dtype=torch.float32)
        if a.is_cuda:
            zeros = zeros.cuda()
        a = torch.cat((a, zeros), 1)
        position = self.get_position_encoding(a.shape[2])
        if a.is_cuda:
            position = position.cuda()
        a[:, -self.position_dims:, :] += position[None, :, :]


        #preembedding
        a = self.beginning(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.p0(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.p1(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.p2(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.p3(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.p4(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.p5(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.p6(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.p7(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.p8(a)
        a[:, -self.position_dims:, :] += position[None, :, :]



        
        #pass the sample through the network
        a = self.layer0(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.layer1(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        a = self.layer2(a)
        a[:, -self.position_dims:, :] += position[None, :, :]
        
        a = self.layer3(a)
        a[:, -self.position_dims:, :] += position[None, :, ::2]
        a = self.layer4(a)
        a[:, -self.position_dims:, :] += position[None, :, ::4]
        a = self.layer5(a)
        a[:, -self.position_dims:, :] += position[None, :, ::8]
        a = self.layer6(a)
        a[:, -self.position_dims:, :] += position[None, :, ::16]
        a = self.layer7(a)
        a[:, -self.position_dims:, :] += position[None, :, ::32]
        a = self.layer8(a)
        a[:, -self.position_dims:, :] += position[None, :, ::64]
        a = self.layer9(a)
        a[:, -self.position_dims:, :] += position[None, :, ::128]


        a = self.layer10(a)
        a[:, -self.position_dims:, :] += position[None, :, ::128]
        a = self.layer11(a)
        a[:, -self.position_dims:, :] += position[None, :, ::128]
        a = self.layer12(a)
        a[:, -self.position_dims:, :] += position[None, :, ::128]

        a = self.f0(a)
        a = self.f1(a)
        a = self.f2(a)
        a = self.f3(a)
        a = self.f4(a)
        a = self.f5(a)
        a = self.f6(a)
        a = self.f7(a)
        a = self.f8(a)
        a = self.final(a)

        return a

    def get_position_encoding(self, length):
        """returns a positional encoding of the specified length"""

        encoding = []
        t = torch.arange(0, length, dtype=torch.float32)

        return torch.stack([torch.sin(2 * pi * t * (i + 1) / 2048) for i in range(self.position_dims)])
        
        # for i in range(1, int(self.position_dims/2)+1):
        #     w = 1 / 2048 ** (i / self.position_dims)
        #     p = torch.sin(w * t * 2 * pi)
        #     pdb.set_trace()
        #     encoding.append(p)
        #     p = torch.cos(w * t * 2 * pi)
        #     encoding.append(p)

        return torch.stack(encoding)



from torch.utils.data import DataLoader
import re

class ModelTrainer():

    def __init__(self, model_directory, model, loader, batch_size=10, epoch_size=100, save_frequency=1, max_epochs=10000000):
        
        self.model = model
        self.model.cuda()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.start_epoch = 0    #epovh numbrt to start at (will be overwritten if model loaded from disk)
        self.i = 0              #index of the number of batches we've run through
        self.epoch = None
        self.epoch_size = epoch_size
        self.max_epochs = max_epochs

        #check if the folder to store the models in exists
        self.model_directory = model_directory
        if not os.path.exists(model_directory):
            os.mkdir(model_directory)

        #check for saved versions of the model on disk
        self.load_idx = self.get_newest_idx()
        if self.load_idx is not None:
            self.load_model()#(model_directory, load_idx, model, optimizer)

        self.model.train() #put the model in training mode

        #set up the data loader and batched data loader
        self.loader = loader
        self.batch_size = batch_size
        self.batch_loader = DataLoader(loader, num_workers=6, shuffle=True, batch_size=batch_size)
        self.batch_loader_iter = None #iterator for the current batch data loader. Reset every epoch

        #keep track of model loss on each batch
        self.loss_history = []

        #set up keeping track of the epochs
        self.save_frequency = save_frequency    #after how many epochs do we save the model
        self.epoch_iter = range(self.start_epoch, 10000000).__iter__()
        # self.new_epoch()

    def __iter__(self):
        return self

    def __next__(self):
        return None


    def new_epoch(self):
        self.epoch = self.epoch_iter.__next__()
        print('Epoch %d -------------------------' % self.epoch)
        if self.epoch != self.start_epoch and self.epoch % self.save_frequency == 0:
            self.save_model()
            self.loss_history = []

        #kick off a new iteration loop for the batch data loader
        self.batch_loader_iter = self.batch_loader.__iter__()
        


    def train(self, train_batch_func, print_loss_func=None):
        if self.i % self.epoch_size == 0 and self.i // self.epoch_size != self.max_epochs:
            self.new_epoch()

        self.model.zero_grad()
        torch.cuda.empty_cache()

        #try to compute the loss. If loss raises StopIteration, start a new epoch and go again
        try:
            loss = train_batch_func(self)
        except StopIteration:
            
            #try to start a new epoch. If new_epoch raises StopIteration, we have completed the final epoch
            try:
                self.new_epoch()
            except StopIteration:
                self.save_model()
                print('Completed Training!\nSaving final model...Done')
                return

            #if new epoch was sucessfully started, compute a new loss 
            loss = train_batch_func(self)

        torch.cuda.empty_cache()

        #take learning step
        loss.backward()
        self.optimizer.step()

        #print out loss for this batch, and update saved iteration parameters
        if print_loss_func is not None:
            print_loss_func(loss.data.item(), self.i)
        else:
            print('%d:  %f' % (self.i, loss.data.item()))
        self.loss_history.append(loss.data.item())
        self.i += 1



    def get_newest_idx(self):
        epochs = [int(filename[len('model_'):]) for filename in os.listdir(self.model_directory) if re.match(r'model_[0-9]+', filename) is not None]
        if len(epochs) == 0:
            return None
        else:
            return max(epochs)

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.model_directory, 'model_%d' % self.load_idx)))
        if self.optimizer is not None:
            self.optimizer.load_state_dict(torch.load(os.path.join(self.model_directory, 'optim_%d' % self.load_idx)))
        
        with open(os.path.join(self.model_directory, 'checkpoint.txt'), 'r') as f:
            nums = [int(line) for line in f]
            self.start_epoch = nums[0]
            self.i = nums[1]
        
        print('Loaded model at epoch %d and iteration %d' % (self.start_epoch, self.i))
        # return model, optimizer, start_epoch, i


    def save_model(self):
        #save model/optimizer
        print('Saving model at epoch %d and iteration %d' % (self.epoch, self.i))
        
        torch.save(self.model.state_dict(), os.path.join(self.model_directory, ('model_%d' % self.epoch)))
        torch.save(self.optimizer.state_dict(), os.path.join(self.model_directory, ('optim_%d' % self.epoch)))
        
        with open(os.path.join(self.model_directory, 'checkpoint.txt'), 'w') as f:
            f.write('%s\n' % str(self.epoch))
            f.write(str(self.i))
        
        with open(os.path.join(self.model_directory, 'loss_history.txt'), 'a') as f:
            for loss in self.loss_history:
                f.write('%s\n' % str(loss))






def train_relative_pitch(trainer):

    #relative pitch (semitones), clip0, clip1
    batch = trainer.batch_loader_iter.__next__()
    dy, x0, x1 = batch

    #move data to GPU
    dy, x0, x1 = dy.cuda(), x0.cuda(), x1.cuda()

    #pass each sample through the same network
    y0 = trainer.model(x0)
    y1 = trainer.model(x1)

    #compute the network's predicted relative pitch
    dy_hat = y1 - y0

    #convert semitones to network pitch scale
    dy = net_semitone_size(dy)

    #create MSE loss for network's relative pitch vs true relative pitch
    error = dy_hat - dy[:, None, None]
    loss = torch.mean(error ** 2)

    del dy, dy_hat, x0, x1, y0, y1, error #free memory on the GPU
    return loss

def train_absolute_pitch(trainer):
    """have the network attempt to predict the absolute pitch of random synthetically generated signals"""
    
    #compute the sizes of the batch we need to make
    batch_size = trainer.batch_size
    rate = trainer.loader.AR
    duration = trainer.loader.duration
    num_samples = int(duration * rate)
    
    #minimum/maximum network activations to bound the minimum/maximum pitches generated
    min_activation=0.25 # ~ 12Hz
    max_activation=1.0  # ~ 22000 Hz

    #create a list of random pitches to generate
    y = torch.tensor(np.random.uniform(min_activation, max_activation, size=(batch_size)))
    pitches = net_pitch_inv(y)

    #create a batch of synthetically generated waves at the specified pitch, and random amplitude
    forms = [waves[idx] for idx in np.random.randint(len(waves), size=(batch_size))]
    x = torch.tensor([[generate_wave(form=form, amplitude=np.random.random()/5, pitch=pitch, duration=duration, FS=rate)] for form, pitch in zip(forms, pitches)], dtype=torch.float32)
    assert(x.shape == torch.Size((batch_size, 1, num_samples)))
    
    #move the generated pitches and waveforms to the GPU
    y, x = y.cuda(), x.cuda()

    #pass the generated waves through the network
    y_hat = trainer.model(x)

    #compute MSE error based on known pitches of generated waves
    error = y_hat - y[:, None, None]
    loss = torch.mean(error ** 2)

    del y, x, y_hat, error
    return loss

def train_synthetic_relative_pitch(trainer):
    """have the network attempt to predict the absolute pitch of random synthetically generated signals"""
    
    #compute the sizes of the batch we need to make
    batch_size = trainer.batch_size
    rate = trainer.loader.AR
    duration = trainer.loader.duration
    num_samples = int(duration * rate)
    
    #minimum/maximum network activations to bound the minimum/maximum pitches generated
    min_activation=0.25 # ~ 12Hz
    max_activation=1.0  # ~ 22000 Hz

    #create a list of random pitches to generate
    y0 = torch.tensor(np.random.uniform(min_activation, max_activation, size=(batch_size)))
    pitches0 = net_pitch_inv(y0)

    y1 = torch.tensor(np.random.uniform(min_activation, max_activation, size=(batch_size)))
    pitches1 = net_pitch_inv(y1)

    #create a batch of synthetically generated waves at the specified pitch, and random amplitude
    forms0 = [waves[idx] for idx in np.random.randint(len(waves), size=(batch_size))]
    x0 = torch.tensor([[generate_wave(form=form, amplitude=np.random.random()/5, pitch=pitch, duration=duration, FS=rate)] for form, pitch in zip(forms0, pitches0)], dtype=torch.float32)

    forms1 = [waves[idx] for idx in np.random.randint(len(waves), size=(batch_size))]
    x1 = torch.tensor([[generate_wave(form=form, amplitude=np.random.random()/5, pitch=pitch, duration=duration, FS=rate)] for form, pitch in zip(forms1, pitches1)], dtype=torch.float32)
    # assert(x0.shape == torch.Size((batch_size, 1, num_samples)))

    #create dy (already in network pitch scale)
    dy = y1 - y0

    #move data to GPU
    dy, x0, x1 = dy.cuda(), x0.cuda(), x1.cuda()

    #pass each sample through the same network
    y0_hat = trainer.model(x0)
    y1_hat = trainer.model(x1)

    #compute the network's predicted relative pitch
    dy_hat = y1_hat - y0_hat

    #create MSE loss for network's relative pitch vs true relative pitch
    error = dy_hat - dy[:, None, None]
    loss = torch.mean(error ** 2)

    del dy, dy_hat, x0, x1, y0, y1, y0_hat, y1_hat, error #free memory on the GPU
    return loss



from math import sqrt
def print_loss_in_semitones(loss, i):
    print(f'{i}: {net_semitone_size_inv(sqrt(loss)).item():.4f} semitones')




import os
from audio_loader import ShiftAudioLoader
#from pitch_detector import AutoPitcherNet
import numpy as np

def train(model_directory=os.path.join('.', 'models')):

    model = AutoPitcherNet()
    loader = ShiftAudioLoader(duration=1.0)
    trainer = ModelTrainer(model_directory, model, loader)

    absolute_frequency = 0.00 #how frequently to inject absolute pitch training examples

    #run the trainer until it finished, or we quit
    for _ in trainer:
        if np.random.random() < absolute_frequency:
            # trainer.train(train_absolute_pitch, print_loss_in_semitones)
            trainer.train(train_synthetic_relative_pitch, print_loss_in_semitones)
        else:
            trainer.train(train_relative_pitch, print_loss_in_semitones)



import soundfile
import matplotlib.pyplot as plt

def test(model_directory=os.path.join('.', 'models')):
    model = AutoPitcherNet()
    loader = ShiftAudioLoader(duration=2)
    trainer = ModelTrainer(model_directory, model, loader) #loads the latest version of the model from disk

    model.eval()
    
    #Load the example from the disk
    paths = [
        './cache/shift_0_-home-david-Programming-MusicScratch-data-VocalSet-male1-arpeggios-straight-m1_arpeggios_straight_a.wav', 
        './cache/shift_0_-home-david-Programming-MusicScratch-data-VocalSet-male2-arpeggios-straight-m2_arpeggios_straight_a.wav',
        './cache/shift_0_-home-david-Programming-MusicScratch-data-VocalSet-male3-arpeggios-straight-m3_arpeggios_straight_a.wav',
        './cache/shift_0_-home-david-Programming-MusicScratch-data-VocalSet-male4-arpeggios-straight-m4_arpeggios_straight_a.wav',
        './cache/shift_0_-home-david-Programming-MusicScratch-data-VocalSet-male5-arpeggios-straight-m5_arpeggios_straight_a.wav'
    ]

    sounds = [soundfile.read(path) for path in paths]
    sounds = [sound for sound, _ in sounds]


    #create syntetic soundwave as well
    # for form in ['sine', 'square', 'square2', 'saw', 'triangle']:
    #     for pitch in [130.8127826502993, 164.81377845643496, 195.99771799087463, 261.6255653005986]:#, 55, 110, 220, 440, 880]:
    #         sounds.append(generate_wave(form=form, amplitude=0.1, pitch=pitch, duration=10, FS=16000))

    #     # x = torch.tensor([[generate_wave(form=form, amplitude=np.random.random(), pitch=pitch, duration=duration, FS=rate)] for form, pitch in zip(forms, pitches)], dtype=torch.float32)



    #combine everything together and plot
    sounds = [torch.tensor([[sound]], dtype=torch.float32).cuda() for sound in sounds]
    
    pitches = []
    for sound in sounds:
        pitch = model(sound)
        pitches.append(pitch[0, 0, 5:-5].detach().cpu())
        del pitch
        torch.cuda.empty_cache()

    # pitches = [model(sound) for sound in sounds]
    # pitches = [pitch[0, 0, 5:-5].detach().cpu() for pitch in pitches]

    for pitch in pitches:
        plt.plot(pitch)

    plt.show()
    pdb.set_trace()






if __name__ == '__main__':
    # while True:
    #     try:
    #         train()
    #     except KeyboardInterrupt:
    #         break
    #     except Exception as e:
    #         print(e)
    # train()
    test()