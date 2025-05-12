import soundfile as sf
import torch
from dia.model import Dia
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # slower, but reproducible
    torch.backends.cudnn.benchmark = False

# Example usage
set_seed(42)

model = Dia.from_local(r"C:\Users\admin\PycharmProjects\dia_finetuning\config.json",r"C:\Users\admin\PycharmProjects\dia_finetuning\checkpoints\ckpt_epoch10.pth")

for param in model.model.parameters():
    param.data = param.data.to(torch.float32)  #
text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

output = model.generate(text=text,max_tokens=4096,top_p=1,temperature=1.3,cfg_filter_top_k=50)

sf.write("simple_2.wav", output, 44100)
