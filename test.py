import os
import sys
import time
import copy
import shutil
import random
import pdb

import torch
import numpy as np
import torchvision.transforms
from tqdm import tqdm
from event_voxel import EventVoxel
import config
from common import myutils
import torchvision.utils as utils
import math
import torch.nn.functional as F

from torch.utils.data import DataLoader
from model.EVFIT_B import UNet_3D_3D




##### Parse CmdLine Arguments #####
os.environ["CUDA_VISIBLE_DEVICES"]='0'
args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)


test_set = EventVoxel(args.data_root, is_hsergb=False, is_training=False, is_validation=False, number_of_time_bins= args.voxel_grid_size)
test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)



print("Building model: %s"%args.model)
model = UNet_3D_3D(n_inputs=args.nbr_frame, joinType=args.joinType)

model = torch.nn.DataParallel(model).to(device)
print("#params" , sum([p.numel() for p in model.parameters()]))

myTransform = torchvision.transforms.ToPILImage()

def save_image(recovery, image_name):
    recovery_image = torch.split(recovery, 1, dim=0)
    batch_num = len(recovery_image)

    if not os.path.exists('./results'):
        os.makedirs('./results')

    for ind in range(batch_num):
        utils.save_image(recovery_image[ind], './results/{}.png'.format(image_name[ind].split("/")))

def to_psnr(rect, gt):
    mse = F.mse_loss(rect, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    psnr_list = [-10.0 * math.log10(mse) for mse in mse_list]
    return psnr_list

def test(args):
    time_taken = []
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()

    with torch.no_grad():
        for i, (images, voxel, gt_image ,paths) in enumerate(tqdm(test_loader)):
            images = [img_.cuda() for img_ in images]
            gt = gt_image.cuda()

            torch.cuda.synchronize()
            start_time = time.time()
            out = model(images, voxel)

            torch.cuda.synchronize()
            time_taken.append(time.time() - start_time)


            save_image(out,paths)
            myutils.eval_metrics(out, gt, psnrs, ssims)

    print("PSNR: %f, SSIM: %fn" %
          (psnrs.avg, ssims.avg))
    print("Time , " , sum(time_taken)/len(time_taken))

    return psnrs.avg


""" Entry Point """
def main(args):
    
    assert args.load_from is not None

    model_dict = model.state_dict()
    model.load_state_dict(torch.load(args.load_from)["state_dict"] , strict=True)
    test(args)


if __name__ == "__main__":
    main(args)
