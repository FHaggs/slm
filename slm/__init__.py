"""Small Language Model. Os ajustes abaixo precisam valer antes de o PyTorch ser importado."""

import os

# Com o padding padrao, o torch.compile tenta alocar ~12 GB em um teste interno e fica sem memoria.
os.environ.setdefault("TORCHINDUCTOR_SHAPE_PADDING", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
