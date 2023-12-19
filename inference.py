import os

import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils import data
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import numpy as np

import argparse
from tqdm import tqdm
import logging

from datasets import get_transforms, get_datasets
from models import build_model, load_model
from config import cfg

from utils import setup_logging
from spikingjelly.clock_driven import functional

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',
                    default='checkpoint/multi_step.pth',
                    help='path to the pretrained checkpoint')
parser.add_argument('--results', default='results/step5', help='location to save predicted saliency maps')

parser.add_argument('--gpu', type=str, default='8', help='gpu id')

parser.add_argument('--reset', default=True, action='store_true', help='reset spiking neuron state')
parser.add_argument('--step', default=False, action='store_true', help='steps')
parser.add_argument('--clip', default=False, action='store_true')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

os.makedirs(args.results, exist_ok=True)
setup_logging(filename=os.path.join(args.results, 'result.txt'))
logger = logging.getLogger(__name__)
if cuda:
    torch.backends.cudnn.benchmark = True
    current_device = torch.cuda.current_device()
    logger.info("Running on " + torch.cuda.get_device_name(current_device))
else:
    logger.info("Running on CPU")

data_mode = cfg.DATA.DATA_MODE
data_transforms = get_transforms(input_size=cfg.DATA.SIZE, image_mode=False)
dataset = get_datasets(name_list=[cfg.DATA.DATA_NAME],
                       split_list=['val'],
                       config_path=cfg.DATA.DATASET_CONFIG,
                       root=cfg.DATA.ROOT,
                       training=False,
                       transforms=data_transforms['val'],
                       read_clip=False,
                       random_reverse_clip=False,
                       clip_len=1,
                       data_mode=data_mode)
dataloader = data.DataLoader(
    dataset=dataset,
    batch_size=1,  # only support 1 video clip
    num_workers=8,
    shuffle=False)

model = build_model(args)

# load pretrained models
if os.path.exists(args.checkpoint):
    logger.info('Loading state dict from: {}'.format(args.checkpoint))
    model = load_model(model=model, model_file=args.checkpoint)
else:
    raise ValueError("Cannot find model file at {}".format(args.checkpoint))

model.to(device)
model = model.eval()


def inference():
    running_mae = 0.0
    running_smean = 0.0
    states = None
    current_video = ''
    for index, data in enumerate(tqdm(dataloader)):
        images = data['image'].unsqueeze(0).to(device)
        video_name = data['image_id'][0].split('/')[0]
        frame_id = data['image_id'][0].split('/')[-1]
        if video_name != current_video:
            current_video = video_name
            states = None
            functional.reset_net(model)
        if args.reset:
            functional.reset_net(model)

        with torch.no_grad():
            preds, middle, _, _ = model(images, states)
            preds = F.sigmoid(preds)

        # save predicted saliency maps
        for j, pred in enumerate(preds):
            dataset = data['dataset'][j]
            image_id = data['image_id'][j]
            result_path = os.path.join(args.results, "{}.png".format(image_id))
            dirname = os.path.dirname(result_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            result = TF.to_pil_image(pred[0][0:1])  # 1,H,W
            result = result.resize((data['width'][j], data['height'][j]))
            result.save(result_path)


if __name__ == "__main__":
    inference()
