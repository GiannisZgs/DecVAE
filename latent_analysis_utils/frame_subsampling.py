import numpy as np
import torch


def frames_per_utterance_index(frame_counts, k):
    """Indices of the frames to keep: at most k frames per utterance, evenly spaced over the utterance.

    frame_counts: number of frames of each utterance, in the order the frames are stored.
    k: frames to keep per utterance; None or 0 keeps every frame. Utterances with k frames or fewer are kept whole.
    Deterministic, so every model keeps the same positions within each utterance.
    """
    frame_counts = np.asarray(frame_counts, dtype=int)
    starts = np.concatenate([[0], np.cumsum(frame_counts)[:-1]])
    if not k:
        return torch.arange(int(frame_counts.sum()))
    keep = [s + (np.arange(n) if n <= k else np.round(np.linspace(0, n - 1, k)).astype(int))
            for s, n in zip(starts, frame_counts)]
    return torch.as_tensor(np.concatenate(keep), dtype=torch.long)
