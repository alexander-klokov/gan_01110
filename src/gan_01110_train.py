import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import random
import numpy

from gan_01110_generator import Generator
from gan_01110_discriminator import Discriminator

N = 20000
DUMMY_INPUT = torch.FloatTensor([0.5])

D = Discriminator()
G = Generator()

label_true = torch.FloatTensor([1.0])
label_false = torch.FloatTensor([0.0])

state = []

def generate_real():
    real_data = torch.FloatTensor(
        [random.uniform(0.0, 0.1),
         random.uniform(0.9, 1.0),
         random.uniform(0.9, 1.0),
         random.uniform(0.9, 1.0),
         random.uniform(0.0, 0.1)])
    return real_data

for j in range(N):
    
    D.train(generate_real(), label_true)
    D.train(G.forward(DUMMY_INPUT).detach(), label_false)
    G.train(D, DUMMY_INPUT, label_true)

    if (j % 1000 == 0):
        state.append( G.forward(torch.FloatTensor([0.9])).detach().numpy() )

    pass

D.save_model()
G.save_model()

# display the progress
D.plot_progress()
G.plot_progress()

plt.show()

# display the state evolution
plt.figure(figsize = (16,8))
plt.imshow(numpy.array(state).T, interpolation='none', cmap='Blues')

plt.show()
