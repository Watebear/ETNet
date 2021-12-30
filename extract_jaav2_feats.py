import argparse
import os
import torch.utils.data as util_data
from tqdm import tqdm

from models.jaav2_feat.jaanet import JaaNetv2
from utils.util import *
from data.data_list import ImageList
from data import pre_process as prep


def main(config):
    use_gpu = torch.cuda.is_available()
    config.use_gpu = use_gpu

    # prepare data
    dsets = {}
    dset_loaders = {}

    dsets['train'] = ImageList(crop_size=config.crop_size, path=config.train_path_prefix,
                               transform=prep.image_train(crop_size=config.crop_size),
                               target_transform=prep.land_transform(img_size=config.crop_size,
                                                                    flip_reflect=np.loadtxt(
                                                                        config.flip_reflect)))

    dset_loaders['train'] = util_data.DataLoader(dsets['train'], batch_size=config.train_batch_size,
                                                 shuffle=True, num_workers=config.num_workers)

    dsets['test'] = ImageList(crop_size=config.crop_size, path=config.test_path_prefix, phase='test',
                              transform=prep.image_test(crop_size=config.crop_size),
                              target_transform=prep.land_transform(img_size=config.crop_size,
                                                                   flip_reflect=np.loadtxt(
                                                                       config.flip_reflect))
                              )
    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config.eval_batch_size,
                                                shuffle=False, num_workers=config.num_workers)

    # set network modules
    net = JaaNetv2(config)
    if use_gpu:
        net = net.cuda()
    net.eval()

    # make tensor save folder
    tensor_save_path = config.train_path_prefix.replace('list', 'tensor_512')
    if not os.path.exists(tensor_save_path):
        os.mkdir(tensor_save_path)

    # extract train feats and save
    loader = dset_loaders['train']
    for i, batch in enumerate(tqdm(loader)):
        input, land, biocular, au = batch
        if use_gpu:
            input, land, au = input.cuda(), land.cuda(), au.cuda()

        aus_output = net(input)
        if i == 0:
            all_output = aus_output.data.cpu().float()
            all_au = au.data.cpu().float()
        else:
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, au.data.cpu().float()), 0)

    all_feat = all_output.data.numpy()
    all_label = all_au.data.numpy()

    train_tensor_save_path = os.path.join(tensor_save_path, 'train')
    if not os.path.exists(train_tensor_save_path):
        os.mkdir(train_tensor_save_path)
    np.save(train_tensor_save_path + '/feats.npy', all_feat)
    np.save(train_tensor_save_path + '/labels.npy', all_label)

    # extract test feats and save
    loader = dset_loaders['test']
    for i, batch in enumerate(tqdm(loader)):
        input, land, biocular, au = batch
        if use_gpu:
            input, land, au = input.cuda(), land.cuda(), au.cuda()

        aus_output = net(input)
        if i == 0:
            all_output = aus_output.data.cpu().float()
            all_au = au.data.cpu().float()
        else:
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, au.data.cpu().float()), 0)

    all_feat = all_output.data.numpy()
    all_label = all_au.data.numpy()

    test_tensor_save_path = os.path.join(tensor_save_path, 'test')
    if not os.path.exists(test_tensor_save_path):
        os.mkdir(test_tensor_save_path)
    np.save(test_tensor_save_path + '/feats.npy', all_feat)
    np.save(test_tensor_save_path + '/labels.npy', all_label)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Misc
    parser.add_argument('--training', type=bool, default=False, help='training or testing')
    parser.add_argument('--use_gpu', type=bool, default=True, help='default use gpu')
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--train_batch_size', type=int, default=80, help='mini-batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=80, help='mini-batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8)

    # Model configuration.
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--map_size', type=int, default=44, help='size for attention maps')
    parser.add_argument('--au_num', type=int, default=8, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_land', type=float, default=0.5, help='weight for landmark detection loss')

    # Directories.
    parser.add_argument('--model_prefix', type=str, default='DISFA_combine_2_3')
    parser.add_argument('--pretrain_prefix', type=str, default='./models/weights/DISFA_combine_2_3')
    parser.add_argument('--train_path_prefix', type=str, default='data/list/DISFA_combine_2_3')
    parser.add_argument('--test_path_prefix', type=str, default='data/list/DISFA_part1')
    parser.add_argument('--flip_reflect', type=str, default='data/list/reflect_49.txt')
    parser.add_argument('--res_path_prefix', type=str, default='results/base_jaa/')

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    main(config)

    # 1_3 -- part2  ;  2_3 -- part1