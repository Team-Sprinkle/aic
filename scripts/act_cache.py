"""Image access for ordinary and sharded verified ACT caches."""
from pathlib import Path

import numpy as np


class ShardedImages:
    def __init__(self, cache, key):
        self.shape = (int(cache["frames"]), *cache["image_shape_hwc"])
        self.shards = []
        self.source_shard_count = len(cache["image_shards"])
        arrays_by_path = {}
        end = 0
        desired_order = cache["image_channel_order"]
        if desired_order not in {"rgb", "bgr"}:
            raise ValueError("Sharded cache needs an audited output channel order")
        for shard in cache["image_shards"]:
            start, stop = shard["from_index"], shard["to_index"]
            source_start = shard.get("source_from_index", 0)
            source_path = (Path(shard["root"]) / (key + ".npy")).resolve()
            if source_path not in arrays_by_path:
                arrays_by_path[source_path] = np.load(source_path, mmap_mode="r")
            array = arrays_by_path[source_path]
            order = shard["image_channel_order"]
            if (start != end or stop <= start or source_start < 0 or source_start + stop - start > len(array)
                    or tuple(array.shape[1:]) != tuple(self.shape[1:]) or array.dtype != np.uint8
                    or order not in {"rgb", "bgr"}):
                raise ValueError(f"Invalid image shard for {key}: {shard}")
            swap = order != desired_order
            # Episode-level manifests may contain hundreds of adjacent slices
            # of the same image array. Reuse one mapping and combine only slices
            # contiguous in BOTH the destination and the original source.
            previous = self.shards[-1] if self.shards else None
            if (previous is not None and previous[1] == start and previous[3] is array
                    and previous[4] == swap and previous[2] + previous[1] - previous[0] == source_start):
                self.shards[-1] = (previous[0], stop, previous[2], array, swap)
            else:
                self.shards.append((start, stop, source_start, array, swap))
            end = stop
        if end != self.shape[0]:
            raise ValueError("Image shards do not cover every cache frame")

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = np.arange(self.shape[0])[index]
        scalar = np.isscalar(index)
        indices = np.atleast_1d(np.asarray(index, dtype=np.int64))
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.shape[0]):
            raise IndexError("ACT cache indices must be a valid one-dimensional frame selection")
        result = np.empty((len(indices), *self.shape[1:]), dtype=np.uint8)
        for start, stop, source_start, array, swap in self.shards:
            selected = (indices >= start) & (indices < stop)
            if selected.any():
                values = array[indices[selected] - start + source_start]
                result[selected] = values[..., ::-1] if swap else values
        return result[0] if scalar else result


def load_cached_images(root, cache):
    if cache.get("image_shards"):
        return {key: ShardedImages(cache, key) for key in cache["camera_keys"]}
    return {key: np.load(Path(root) / (key + ".npy"), mmap_mode="r") for key in cache["camera_keys"]}


def prepare_camera_arrays(images, indices, executor=None):
    """Yield camera batches in canonical order; worker threads only touch CPU data."""
    def read(image):
        return np.ascontiguousarray(image[indices].transpose(0, 3, 1, 2))
    if executor is None:
        for key, image in images.items():
            yield key, read(image)
    else:
        futures = [(key, executor.submit(read, image)) for key, image in images.items()]
        for key, future in futures:
            yield key, future.result()
