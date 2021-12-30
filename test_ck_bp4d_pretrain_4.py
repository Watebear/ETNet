import argparse
import os
import torch.utils.data as util_data
from tqdm import tqdm

from models.jaav2_base.jaanet import JaaNetv2
from utils.util import *
from data_ck.data_list import ImageList
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

    # save result
    if not os.path.exists(config.res_path_prefix):
        os.mkdir(config.res_path_prefix)
    if not os.path.exists(config.res_path_prefix + config.model_prefix):
        os.mkdir(config.res_path_prefix + config.model_prefix)
    res_file = open(config.res_path_prefix + config.model_prefix + '/result.txt', 'w')
    au_pred_file = config.res_path_prefix + config.model_prefix + '/au_pred.txt'

    loader = dset_loaders['test']
    for i, batch in enumerate(tqdm(loader)):
        input, land, biocular, au = batch
        if use_gpu:
            input, land, au = input.cuda(), land.cuda(), au.cuda()

        aus_output, local_aus_output, align_output = net(input)
        if i == 0:
            all_local_output = local_aus_output.data.cpu().float()
            all_output = aus_output.data.cpu().float()
            all_au = au.data.cpu().float()
            all_pred_land = align_output.data.cpu().float()
            all_land = land.data.cpu().float()
        else:
            all_local_output = torch.cat((all_local_output, local_aus_output.data.cpu().float()), 0)
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, au.data.cpu().float()), 0)
            all_pred_land = torch.cat((all_pred_land, align_output.data.cpu().float()), 0)
            all_land = torch.cat((all_land, land.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    local_AUoccur_pred_prob = all_local_output.data.numpy()
    AUoccur_actual = all_au.data.numpy()
    pred_land = all_pred_land.data.numpy()
    GT_land = all_land.data.numpy()

    # save AUoccur_pred_prob
    np.savetxt(au_pred_file, AUoccur_pred_prob, fmt='%f', delimiter='\t')

    local_f1score_arr, local_acc_arr, f1score_arr, acc_arr, mean_error, failure_rate = au_detection_eval_ck_bp4d(
        AUoccur_pred_prob, local_AUoccur_pred_prob, AUoccur_actual, pred_land, GT_land, keep=[0, 1, 3, 6], actual_keep=[0, 1, 2, 4]
    )
    #print('=============================')
    #print(local_f1score_arr.shape)
    #print(local_acc_arr.shape)
    #print(f1score_arr.shape)
    #print(acc_arr.shape)
    #print(mean_error.shape)
    #print(failure_rate.shape)

    # au_detection_eval_ck_bp4d
    # bp4d: 1,2,4,6,7,10,12,14,15,17,23,24
    # ck+ : 1,2,-,6,7,--,12,14,15,17,23,24  # del pred of [:, 2] and [:, 5]
    # actual: 1, 2, 6, 7, 12, 15, 17, 23, 24
    # record result
    line1 = "Test model, train on {}, test on data: {}".format(config.model_prefix, config.test_path_prefix)
    line2 = "F1 score of each au is: " \
            "au1={},  au2={},  au6={},  au12={}" .format(
        f1score_arr[0], f1score_arr[1], f1score_arr[2], f1score_arr[3])
    line3 = "Avarage F1 score is: avg={}".format(f1score_arr.mean())
    line4 = "Local F1 score of each au is: " \
            "au1={},  au2={},  au6={},  au12={}".format(
        local_f1score_arr[0], local_f1score_arr[1], local_f1score_arr[2], local_f1score_arr[3])
    line5 = "Local Avarage F1 score is: avg={}".format(local_f1score_arr.mean())
    line6 = "Landmark mean error is: mean_error= {}".format(mean_error)
    line7 = "Acc of each au is: " \
            "au1={},  au2={},  au6={}, au12={}".format(
        acc_arr[0], acc_arr[1], acc_arr[2], acc_arr[3])
    line8 = "Local acc is: local_acc_arr={}, Acc is: acc_arr={}".format(local_acc_arr.mean(),
                                                                        acc_arr.mean())

    for line in [line1, line2, line3, line4, line5, line6, line7, line8]:
        print(line+'\n')
        res_file.write(line+'\n')
    res_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Misc
    parser.add_argument('--training', type=bool, default=False, help='training or testing')
    parser.add_argument('--use_gpu', type=bool, default=True, help='default use gpu')
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='mini-batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8)

    # Model configuration.
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--map_size', type=int, default=44, help='size for attention maps')
    parser.add_argument('--au_num', type=int, default=12, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_land', type=float, default=0.5, help='weight for landmark detection loss')

    # Directories.
    parser.add_argument('--model_prefix', type=str, default='CK_BP4D_combine_1_2')
    parser.add_argument('--pretrain_prefix', type=str, default='./models/weights/BP4D_combine_1_2')
    parser.add_argument('--train_path_prefix', type=str, default='./data_ck/lists/CK')
    parser.add_argument('--test_path_prefix', type=str, default='./data_ck/lists/CK')
    parser.add_argument('--flip_reflect', type=str, default='data/list/reflect_49.txt')
    parser.add_argument('--res_path_prefix', type=str, default='results/CK_BP4D_combine1_2/')

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    main(config)