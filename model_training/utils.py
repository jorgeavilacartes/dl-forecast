import json
import logging
import importlib
from pathlib import Path
from collections import Counter

def asJSON(data, path_save, sort_keys):
    with open(str(path_save), 'w', encoding='utf8') as fp:
        json.dump(data,fp, ensure_ascii=False, indent=4, sort_keys=sort_keys)

def fromJSON(json_file):
    """load json with utf8 encoding"""
    with open(str(json_file),"r", encoding='utf8') as fp:
        return json.load(fp)

def set_logger(*, path_log: str = ""):
    path_log = path_log if path_log.endswith(".log") else path_log + ".log"
    path_log = Path(path_log)
    # Create and configure logger
    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename =path_log,
                        level    = logging.DEBUG,
                        format   = LOG_FORMAT,
                        filemode = 'w'
                        )
    logger = logging.getLogger()
    return logger

# def get_count_labels(list_paths, labels, encode_label):
#     """Get frequency per class in one dataset

#     Args:
#         list_img_paths (list): list with path to images of one dataset 
#         labels (dict): dictionary with path to images as keys and list with labels as values

#     Returns:
#         dict: frequency per class
#     """    
#     list_labels = []
#     for img in list_paths:
#         list_labels.extend( [encode_label[label] for label in labels[img]] )
#     return dict(Counter(list_labels))