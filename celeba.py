#%%
from torchvision import datasets

from loaddata import LoadData

dataset = datasets.CelebA
train_data = dataset('../input/data', download = True)
# %%
train_data.shape
