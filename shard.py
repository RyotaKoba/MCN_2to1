from tqdm.auto import tqdm
import numpy as np
import webdataset as wds
import os
import random

dataset_root = './data/howto100M/HowTo100M_1166_videopath890000.txt'
dir_path = "/home/kobayashi/video"

with open(dataset_root, "r") as file:
    file_paths = [os.path.join(dir_path, line.strip() + ".npz") for i, line in enumerate(file) if i != 0]
random.shuffle(file_paths)

shard_path = '/home/kobayashi/shard/video_01'

shard_dir_path = os.pardir(shard_path)
shard_filename = os.path.join(shard_dir_path,f"shards_video.tar")

# shard_size = int(50 * 1000**2)  # 50MB each

with wds.ShardWriter(shard_filename) as sink, tqdm(file_paths) as pbar:
    for file_path in pbar:
        _,_,_,_,_,video_id = file_path.strip().split("/")
        key_str = video_id.rstrip('.npz')
        sink.write({
            "__key__": key_str,
            "npz": np.load(file_path)
        })
        