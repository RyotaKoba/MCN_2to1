from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import json
import webdataset as wds
import os
import random

dataset_root = './data/howto100M/HowTo100M_1166_videopath890000.txt'

file_paths = [
    path for path in Path(dataset_root).glob('*/*')
    if not path.is_dir() 
        and path.name.endswith((
            '.JPEG', '.jpeg', '.jpg',
        ))
]
random.shuffle(file_paths)

print(file_paths[:2])

category_list = sorted([
    path.name for path in Path(dataset_root).glob('*') if path.is_dir()
    ])
category_index = {
    category_name: i 
    for i, category_name in enumerate(category_list)
    }

shard_path = './data/shards_01'
dataset_root = "/home/video/"

shard_dir_path = os.pardir(shard_path)
shard_filename = os.path.join(shard_dir_path,f"shards_video.tar")

# shard_size = int(50 * 1000**2)  # 50MB each

with wds.ShardWriter(shard_filename) as sink, tqdm(file_paths) as pbar:

    for file_path in pbar:
        category_name = file_path.parent.name
        label = category_index[category_name]
        key_str = category_name + '/' + file_path.stem

        sink.write({
            "__key__": key_str,
            "jpg": np.array(Image.open(file_path)),
            "cls": label,
        })

dataset_size = len(shard_filename)

dataset_size_filename = str(
    shard_dir_path / 'dataset-size.json')
with open(dataset_size_filename, 'w') as fp:
    json.dump({
        "dataset size": dataset_size,
        "n_classes": len(category_index),
    }, fp)
