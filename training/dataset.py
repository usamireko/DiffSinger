import pathlib

import numpy
import torch

from utils import collate_nd
from utils.indexed_datasets import IndexedDataset


class BaseDataset(torch.utils.data.Dataset):
    __non_zero_paddings__ = {
        "uv": True, "note_midi": -1, "note_rest": True,
    }

    def __init__(self, binary_data_dir: pathlib.Path, prefix: str):
        super().__init__()
        self.info = {
            k: v
            for k, v in numpy.load(binary_data_dir / f"{prefix}.info.npz").items()
        }
        self.data = IndexedDataset(binary_data_dir, prefix)
        self.epoch = 0

    def __getitem__(self, index):
        return {"_idx": index, **self.data[index]}

    def __len__(self):
        return self.info["lengths"].shape[0]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def num_frames(self, index: int) -> int:
        return self.info["lengths"][index]

    @classmethod
    def collate(cls, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = {
            "size": len(samples),
            "indices": torch.LongTensor([s.pop("_idx") for s in samples]),
        }
        if len(samples) == 0:
            return batch
        for key, value in samples[0].items():
            if value.ndim == 0:
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                pad_value = cls.__non_zero_paddings__.get(key, 0)
                batch[key] = collate_nd([s[key] for s in samples], pad_value=pad_value)
        return batch


class DynamicBatchSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(
            self,
            dataset: BaseDataset,
            max_batch_size: int,
            max_batch_frames: int,
            sort_by_len: bool = True,
            frame_count_grid: int = 1,
            batch_count_multiple_of: int = 1,
            reassign_batches: bool = True,
            shuffle_batches: bool = True,
            seed: int = 0,
    ):
        if torch.distributed.is_initialized():
            num_replicas = None
            rank = None
        else:
            num_replicas = 1
            rank = 0
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_batch_frames = max_batch_frames
        self.sort_by_len = sort_by_len
        self.frame_count_grid = frame_count_grid
        self.batch_count_multiple_of = batch_count_multiple_of
        self.reassign_batches = reassign_batches
        self.shuffle_batches = shuffle_batches
        self.generator: torch.Generator = torch.Generator().manual_seed(seed)
        self.batches: list[list[int]] = None
        self.formed = None

    def __iter__(self):
        self.form_batches()
        return iter(self.batches)

    def __len__(self):
        self.form_batches()
        return len(self.batches)

    def set_epoch(self, epoch: int):
        super().set_epoch(epoch)
        self.generator = torch.Generator().manual_seed(self.seed + epoch)
        self.dataset.set_epoch(epoch)

    def permutation(self, n: int) -> list[int]:
        perm = torch.randperm(n, generator=self.generator).tolist()
        return perm

    def form_batches(self):
        if self.formed == self.epoch + self.seed:
            return

        lengths = [self.dataset.num_frames(i) for i in range(len(self.dataset))]
        if self.sort_by_len:
            sorted_indices = sorted(
                self.permutation(len(lengths)),
                key=lambda x: lengths[x] // self.frame_count_grid, reverse=True
            )
        else:
            sorted_indices = list(range(len(lengths)))

        batches: list[list[int]] = []
        current_batch = []
        current_frames = 0

        for idx in sorted_indices:
            sample_length = lengths[idx]
            if sample_length > self.max_batch_frames:
                raise ValueError(
                    f"Sample length {sample_length} exceeds max batch frames {self.max_batch_frames}."
                )
            if (len(current_batch) >= self.max_batch_size or
                    current_frames + sample_length > self.max_batch_frames):
                batches.append(current_batch)
                current_batch = []
                current_frames = 0
            current_batch.append(idx)
            current_frames += sample_length
        if current_batch:
            batches.append(current_batch)

        multiple_of = self.num_replicas * self.batch_count_multiple_of
        remainder = (multiple_of - (len(batches) % multiple_of)) % multiple_of
        if self.reassign_batches:
            new_batch = []
            new_batch_frames = 0
            while remainder > 0:
                num_batches = len(batches)
                perm = self.permutation(num_batches)
                batches = [batches[i] for i in perm]
                modified = False
                idx = 0
                while remainder > 0 and idx < num_batches:
                    batch = batches[idx]
                    if len(batch) > 1:
                        item = batch[-1]
                        if (len(new_batch) == self.max_batch_size or
                                new_batch_frames + lengths[item] > self.max_batch_frames):
                            batches.append(new_batch)
                            new_batch = []
                            new_batch_frames = 0
                            modified = True
                            remainder -= 1
                            if remainder == 0:
                                break
                        batch.pop()
                        new_batch.append(item)
                        new_batch_frames += lengths[item]
                    idx += 1
                if not modified:
                    if len(new_batch) > 0:
                        batches.append(new_batch)
                        new_batch = []
                        new_batch_frames = 0
                        remainder -= 1
                    if remainder > 0:
                        raise RuntimeError(
                            f"Unable to reassign batches to meet the required multiple count of {multiple_of}."
                        )
        else:
            batches += [[]] * remainder

        if self.shuffle_batches:
            perm = self.permutation(len(batches))
            batches = [batches[i] for i in perm]
        elif self.sort_by_len:
            batches = sorted(
                batches,
                key=lambda b: sum(self.dataset.num_frames(i) for i in b),
                reverse=True
            )

        batches = [b for i, b in enumerate(batches) if i % self.num_replicas == self.rank]

        self.batches = batches
        self.formed = self.epoch + self.seed
