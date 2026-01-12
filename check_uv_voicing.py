import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy

from utils.indexed_datasets import IndexedDataset

data_file = pathlib.Path(r"data/qixuan_v4/binary/acoustic/train.data")
meta_file = pathlib.Path(r"data/qixuan_v4/binary/acoustic/train.meta")
save_dir = pathlib.Path(r"data/_check")
save_dir.mkdir(parents=True, exist_ok=True)
threshold = -30
with open(meta_file, 'rb') as f:
    metadata = pickle.load(f)
data = IndexedDataset(data_file.parent, data_file.stem)
for i, name in enumerate(metadata["names"]):
    spk = metadata["spk_names"][i]
    sample = data[i]
    if not sample["uv"].any():
        continue
    uv_voicing_max = sample["voicing"][sample["uv"]].max()
    if uv_voicing_max < threshold:
        continue
    print(f"{i:06d}_{spk}_{name}: max = {uv_voicing_max}")
    mel = sample["mel"].cpu().numpy().T
    uv = sample["uv"][None].repeat(mel.shape[0], 1).cpu().numpy()
    alpha = numpy.where(uv, 0.2, 0)
    voicing = sample["voicing"].cpu().numpy()

    fig = plt.figure(figsize=(24, 6))
    plt.pcolor(mel, vmin=-14, vmax=4)
    plt.pcolor(uv, alpha=alpha, color='red')
    plt.plot((voicing / 96 + 1) * 128, label="voicing", color="white")
    plt.plot(numpy.full_like(voicing, (threshold / 96 + 1) * 128),
             label=f"threshold={threshold:.2f}", color="yellow", linestyle="--")
    plt.ylim(0, mel.shape[0])
    plt.title(f"{spk} - {name}: max uv voicing = {uv_voicing_max:.2f}")
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_dir / f"{i:06d}_{spk}_{name}_uv_voicing.jpg")
    plt.close(fig)

del data
