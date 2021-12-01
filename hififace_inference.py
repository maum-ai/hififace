import os
import math
import argparse
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from hififace_pl import HifiFace
from dataset import HifiFaceParsingValDataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=int, default=-1)

parser.add_argument('--model_config', type=str, default='config/model.yaml')

parser.add_argument('--model_checkpoint_path', type=str, required=True)

parser.add_argument('--input_directory_path', type=str, default=None)

parser.add_argument('--source_image_path', type=str, default=None)
parser.add_argument('--source_image_path2', type=str, default=None)
parser.add_argument('--target_image_path', type=str, default=None)
parser.add_argument('--output_image_path', type=str, default=None)
parser.add_argument('--interpolation_identity', action='store_true')
parser.add_argument('--interpolation_3d', action='store_true')
parser.add_argument('--interpolation_all', action='store_true')
args = parser.parse_args()

device = torch.device('cpu') if args.gpus == -1 else torch.device(f'cuda:{args.gpus}')

net = HifiFace(OmegaConf.load(args.model_config))
checkpoint = torch.load(args.model_checkpoint_path, map_location='cpu')
net.load_state_dict(checkpoint['state_dict'])
net.eval()
net.to(device)

pil2tensor_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.ToTensor()])
tensor2pil_transform = transforms.ToPILImage()

if args.source_image_path is not None and args.target_image_path is not None:
    target_img = Image.open(args.target_image_path)
    target_img = target_img.convert('RGB')
    target_img = pil2tensor_transform(target_img)
    target_img = target_img.unsqueeze(0)
    target_img = target_img.to(device)

    source_img = Image.open(args.source_image_path)
    source_img = source_img.convert('RGB')
    source_img = pil2tensor_transform(source_img)
    source_img = source_img.unsqueeze(0)
    source_img = source_img.to(device)

    if args.interpolation_all or args.interpolation_identity or args.interpolation_3d:
        if args.source_image_path2 is not None:
            source_img2 = Image.open(args.source_image_path2)
            source_img2 = source_img2.convert('RGB')
            source_img2 = pil2tensor_transform(source_img2)
            source_img2 = source_img2.unsqueeze(0)
            source_img2 = source_img2.to(device)
        else:
            source_img2 = target_img.clone().detach()

        mode = ''
        if args.interpolation_all:
            mode = 'all'
        elif args.interpolation_identity:
            mode = 'identity'
        elif args.interpolation_3d:
            mode = '3d'

        output_img_list = []
        for i in tqdm(range(21)):
            i_ = i / 20.0
            with torch.no_grad():
                output_img = net.interp(source_img, source_img2, target_img, i_, mode)
            output_img = output_img.cpu().clamp(0, 1).squeeze()
            output_img = tensor2pil_transform(output_img)
            output_img_list.append(output_img)

        output_img_list[0].save(args.output_image_path, format='GIF',
                       append_images=output_img_list[1:],
                       save_all=True,
                       duration=300, loop=0)

    else:
        with torch.no_grad():
            output_img = net(source_img, target_img)
        output_img = output_img.cpu().clamp(0, 1).squeeze()
        output_img = tensor2pil_transform(output_img)
        output_img.save(args.output_image_path)

elif args.input_directory_path is not None and args.args.output_image_path is not None:
    dataset = HifiFaceParsingValDataset(img_root=args.input_directory_path, parsing_root=args.input_directory_path)
    dataloader = DataLoader(dataset, batch_size=1)

    image_list = []
    for idx, batch in tqdm(enumerate(dataloader)):
        target_img = batch['target_image']
        target_img = target_img.to(device)

        source_img = batch['source_image']
        source_img = source_img.to(device)

        with torch.no_grad():
            output_img = net(source_img, target_img)
        output_img = output_img.cpu().clamp(0, 1).squeeze()
        image_list.append(output_img)

    grid_image_tensor = make_grid(image_list, nrow=int(math.sqrt(len(image_list))))
    grid_image = tensor2pil_transform(grid_image_tensor)
    grid_image.save(args.output_image_path)