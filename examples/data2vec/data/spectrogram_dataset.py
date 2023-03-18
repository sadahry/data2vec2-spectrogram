# Based on data2vec 2.0 https://github.com/facebookresearch/fairseq/blob/d871f6169f8185837d1c11fb28da56abfd83841c/fairseq/data/audio/raw_audio_dataset.py
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import time
import numpy as np
import torch
from fairseq.data import FileAudioDataset


logger = logging.getLogger(__name__)


class FileSpectrogramDataset(FileAudioDataset):
    def __init__(
        self,
        manifest_path,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=0,  # will not use sample_rate
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
        )

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        source_dim = None
        collated_sources = sources[0].new_zeros(
            len(sources), target_size, *sources[0].shape[1:]
        )
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            # get channel dimension of source
            _, C = source.shape
            source_dim = C if source_dim is None else source_dim
            assert source_dim == C, "source dim shuld be the same between batches"

            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff, *source.shape[1:]), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        out["net_input"] = input
        return out

    def __getitem__(self, index):
        fname = self.fnames[index]
        fname = self.text_compressor.decompress(fname)
        fpath = os.path.join(self.root_dir, fname)

        retry = 3
        feats = None
        for i in range(retry):
            try:
                feats = np.load(fpath)
                break
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}. Sleeping for {1 * i}")
                time.sleep(1 * i)

        if feats is None:
            raise Exception(f"Failed to load {fpath}")

        feats = torch.from_numpy(feats).float()

        v = {"id": index, "source": feats}

        return v
