from .blender import BlenderDataset
from .llff import LLFFDataset
from .phototourism import PhototourismDataset
from .tanks_and_temples import TanksAndTemplesDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'phototourism': PhototourismDataset,
                't&t': TanksAndTemplesDataset,
                }