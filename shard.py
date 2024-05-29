from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import json

shard_path = './shards_01'

shard_dir_path = Path(shard_path)
shard_dir_path.mkdir(exist_ok=True)
shard_filename = str(shard_dir_path / 'shards-%05d.tar')

shard_size = int(50 * 1000**2)  # 50MB each

with wds.ShardWriter(
    shard_filename,
    maxsize=shard_size,
    ) as sink, tqdm(
        file_paths
    ) as pbar:

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