from .blender import BlenderDataset
from .llff import LLFFDataset
from .tanks_and_temples import TanksAndTemplesDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                't&t': TanksAndTemplesDataset}