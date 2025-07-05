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
tf_stub.Tensor = object
tf_stub.TensorShape = object
tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
sys.modules.setdefault("tensorflow", tf_stub)

from transformers import CLIPProcessor, CLIPModel

_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()

@torch.no_grad()
def clip_similarity(img_BCHW, prompt_list):
    pil_imgs = [TF.to_pil_image(img) if torch.is_tensor(img) else Image.open(img) for img in img_BCHW]
    ipt = _proc(text=prompt_list, images=pil_imgs, return_tensors="pt", padding=True).to("cuda")
    out = _model(**ipt)
    return torch.cosine_similarity(out.image_embeds, out.text_embeds)
