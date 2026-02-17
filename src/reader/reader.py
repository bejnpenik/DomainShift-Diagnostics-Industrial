import numpy as np

from scipy.io import loadmat
from pathlib import Path
from collection import Metadata


class BaseFileReader:
    pass

class CWRUFileReader(BaseFileReader):
    
    def __call__(self, fname:str, metadata:Metadata)->tuple[np.ndarray, dict]:


        bearing = metadata['bearing_position']

        data = loadmat(fname, appendmat=True)

        key = Path(fname).name.split('.')[0]

        if key == '101':
            key = '97'
        elif key == '102':
            key = '98'
        elif key == '103':
            key = '99'
        elif key == '104':
            key = '100'

        elif key == '174':
            key = '173'

        if len(key) < 3:
            key = f'0{key}'

        key = f'X{key}_{bearing}_time'
        

        return np.asarray(data[key].ravel(), dtype=np.float32)

class PaderbornFileReader(BaseFileReader):
    def __call__(self, fname: str, metadata:Metadata|None=None, variable:int=6):

        data_dict = loadmat(fname, simplify_cells=True, appendmat=True)
        
        dict_key = Path(fname).name.split('.')[0].split('_')

        dict_key = '_'.join(dict_key)

        x = data_dict[dict_key]['Y'][variable]['Data']
        
        return np.asarray(x, dtype=np.float32)

