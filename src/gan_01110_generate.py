import torch
import matplotlib.pyplot as plt

from gan_01110_generator import Generator

DUMMY_INPUT = torch.FloatTensor([0.5])

G = Generator()
G.load_model()

data_generated = G.forward(DUMMY_INPUT).detach().numpy()

plt.bar([0,1,2,3,4], height=data_generated)
plt.show()
