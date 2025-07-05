# -*- coding: utf-8 -*-
import os
import types
import sys
import importlib.machinery
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Prevent transformers from importing TensorFlow. Some environments ship with
# a minimal or incompatible TensorFlow stub which breaks the runtime when
# transformers attempts to inspect it.  Setting the environment variable and
# providing our own lightweight stub avoids any TensorFlow interaction.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

tf_stub = types.ModuleType("tensorflow")

class _DummyTensor:
    pass

class _DummyTensorShape:
    pass

tf_stub.Tensor = _DummyTensor
tf_stub.TensorShape = _DummyTensorShape
tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
sys.modules.setdefault("tensorflow", tf_stub)

from transformers import CLIPProcessor, CLIPModel

_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_model = None


def _get_model():
    """Lazily create the CLIP model on the first call."""
    global _model
    if _model is None:
        _model = (
            CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            .cuda()
            .eval()
        )
    return _model

@torch.no_grad()
def clip_similarity(img_BCHW, prompt_list):
    pil_imgs = [TF.to_pil_image(img) if torch.is_tensor(img) else Image.open(img) for img in img_BCHW]
    ipt = _proc(text=prompt_list,
                images=pil_imgs,
                return_tensors="pt",
                padding=True,
                truncation=True).to("cuda")
    model = _get_model()
    out = model(**ipt)
    return torch.cosine_similarity(out.image_embeds, out.text_embeds)
