# Based on data2vec 2.0 https://github.com/facebookresearch/fairseq/blob/d871f6169f8185837d1c11fb28da56abfd83841c/fairseq/data/audio/raw_audio_dataset.py
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import time
import numpy as np
from fairseq.data import FileAudioDataset

logger = logging.getLogger(__name__)


class SpectrogramDataset(FileAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
        )

    def __getitem__(self, index):
        fname = self.fnames[index]
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

        return feats
