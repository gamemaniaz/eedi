import torch


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")