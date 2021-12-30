import argparse
import os
import torch.utils.data as util_data
from tqdm import tqdm

from models.jaav2_base_ck_feat import JaaNetv2
from utils.util import *
from data_ck.data_list_dk_feat import ImageList
from data import pre_process as prep


def main(config):
    use_gpu = torch.cuda.is_available()
    config.use_gpu = use_gpu

    # prepare data
    dsets = {}
    dset_loaders = {}

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
    loader = dset_loaders['test']
    for i, batch in enumerate(tqdm(loader)):
        input, path = batch
        if use_gpu:
            input = input.cuda()

        aus_output = net(input)
        sub, seq, im_pa = path[0].split('/')
        sub_path = os.path.join(config.feat_path, sub)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
        seq_path = os.path.join(sub_path, seq)
        if not os.path.exists(seq_path):
            os.mkdir(seq_path)
        feat_name = im_pa.replace(".png", ".npy")
        feat_name = os.path.join(seq_path, feat_name)

        aus_output = aus_output.cpu().detach().numpy()[0]
        np.save(feat_name, aus_output)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Misc
    parser.add_argument('--training', type=bool, default=False, help='training or testing')
    parser.add_argument('--use_gpu', type=bool, default=True, help='default use gpu')
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--train_batch_size', type=int, default=1, help='mini-batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='mini-batch size for evaluation')
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
    parser.add_argument('--model_prefix', type=str, default='DISFA_combine_1_3')
    parser.add_argument('--pretrain_prefix', type=str, default='./models/weights/DISFA_combine_1_3')
    parser.add_argument('--train_path_prefix', type=str, default='data/list/DISFA_combine_1_3')
    parser.add_argument('--test_path_prefix', type=str, default='./data_ck/lists/CK')
    parser.add_argument('--flip_reflect', type=str, default='data/list/reflect_49.txt')
    parser.add_argument('--res_path_prefix', type=str, default='results/CK_DISFA_combine1_3/')
    parser.add_argument('--feat_path', type=str, default='data_ck/feats')

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    main(config)

    # 1_3 -- part2  ;  2_3 -- part1