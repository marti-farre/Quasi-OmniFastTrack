import torch
import os
import argparse
import re
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--seq_name', type=str, required=True, help='sequence name')
parser.add_argument('--data_dir', type=str, default = 'data/mnt/data/data_release/', required=False, help='sequence name')

args = parser.parse_args()

# ZoeD_NK
conf = get_config("zoedepth_nk", "infer")
model_zoe_nk = build_model(conf)


zoe = model_zoe_nk.to(DEVICE)
path_omni = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(path_omni, args.data_dir, args.seq_name)

# Local file
for filename in os.listdir(os.path.join(data_path, "color")):
    image = Image.open(os.path.join(data_path, "color/{}".format(filename))).convert("RGB")

    depth = zoe.infer_pil(image)
    colored = colorize(depth)

    ids = re.sub(r'\D', '', filename)
    print(ids)
    fpath_colored = os.path.join(data_path, "depths/depth_{}.png".format(ids))
    if not os.path.exists(os.path.join(data_path, "depths")):
        os.makedirs(os.path.join(data_path, "depths"))
    Image.fromarray(colored).save(fpath_colored)
