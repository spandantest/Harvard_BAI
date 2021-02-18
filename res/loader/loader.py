import pickle
import pathlib
current_path = pathlib.Path(__file__).parent.absolute()
print('Loaders are being loaded from: %s'%current_path)
import sys
sys.path.append(current_path)
# from loader.multi_attribute_loader_file_list import FileListFolder as multi_attribute_loader_file_list

from .cats_dogs_loader import FileListFolder

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cats_dogs_loader" : FileListFolder
    }[name]