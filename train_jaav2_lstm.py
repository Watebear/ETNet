import argparse
import os
import torch.optim as optim
import torch.utils.data as util_data
from loguru import logger
import time

from models.jaav2_lstm import JaaNetv2_lstm
from utils.util import *
from data.data_list_5 import ImageList_5
from data import pre_process as prep
from utils import lr_schedule

optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}

def main(config):
    use_gpu = torch.cuda.is_available()
    config.use_gpu = use_gpu

    ## prepare data
    dsets = {}
    dset_loaders = {}

    dsets['train'] = ImageList_5(crop_size=config.crop_size, path=config.train_path_prefix,
                                 transform=prep.image_train(crop_size=config.crop_size),
                                 target_transform=prep.land_transform(img_size=config.crop_size,
                                                                      flip_reflect=np.loadtxt(
                                                                          config.flip_reflect)))

    dset_loaders['train'] = util_data.DataLoader(dsets['train'], batch_size=config.train_batch_size,
                                                 shuffle=True, num_workers=config.num_workers)

    dsets['test'] = ImageList_5(crop_size=config.crop_size, path=config.test_path_prefix, phase='test',
                                transform=prep.image_test(crop_size=config.crop_size),
                                target_transform=prep.land_transform(img_size=config.crop_size,
                                                                     flip_reflect=np.loadtxt(
                                                                         config.flip_reflect))
                                )

    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config.eval_batch_size,
                                                shuffle=False, num_workers=config.num_workers)

    # set network modules
    net = JaaNetv2_lstm(config)
    if config.resume_model:
        ckpt = torch.load(config.resume_model)
        net.load_state_dict(ckpt, strict=True)
    if use_gpu:
        net = net.cuda()

    ## set optimizer: SGD
    optimizer = optim_dict[config.optimizer_type](
        net.parameters(),
        lr=1.0, momentum=config.momentum, weight_decay=config.weight_decay,
        nesterov=config.use_nesterov)

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])

    lr_scheduler = lr_schedule.schedule_dict[config.lr_type]

    ## eval result file
    res_file = open(config.task_log_prefix + '/eval_result.txt', 'w')

    ## train
    net.train()
    for epoch in range(config.start_epoch, config.n_epochs+1):
        logger.info("\n")
        logger.info("** Start Epoch {} **".format(epoch))
        # train
        net.train()
        net.training = True
        for i, batch in enumerate(dset_loaders['train']):
            input, land, biocular, au = batch
            if use_gpu:
                input = input.cuda()
                land = land.float().cuda()
                biocular = biocular.float().cuda()
                au = au.long().cuda()
            else:
                au = [a.long() for a in au]

            # adjust
            optimizer = lr_scheduler(param_lr, optimizer, epoch, config.gamma, config.stepsize, config.init_lr)
            optimizer.zero_grad()
            # forward
            total_loss, loss_au_softmax, loss_au_dice, loss_local_au_softmax, \
            loss_local_au_dice, loss_land = net(input, au, land, biocular)
            # backward
            total_loss.backward()
            optimizer.step()

            if i > 0 and i % config.print_freq == 0:
                line_l = "epoch={} || iter={} || total_loss={:.4f} ||  loss_au_softmax={:.4f} || loss_au_dice={:.4f} || " \
                        "loss_local_au_softmax={:.4f} || loss_local_au_dice={:.4f} || loss_land={:.4f} || \
                         learning_rate={} \n".format(epoch, i, total_loss.data.cpu().numpy(), loss_au_softmax.data.cpu().numpy(),
                                            loss_au_dice.data.cpu().numpy(), loss_local_au_softmax.data.cpu().numpy(),
                                            loss_local_au_dice.data.cpu().numpy(), loss_land.data.cpu().numpy(),
                                            optimizer.param_groups[0]['lr'])
                logger.info(line_l)

        # eval
        net.eval()
        net.training = False
        # each batch
        for i, batch in enumerate(dset_loaders['test']):
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
        au_pred_file = config.task_log_prefix + '/Epoch{}_au_pred.txt'.format(epoch)
        np.savetxt(au_pred_file, AUoccur_pred_prob, fmt='%f', delimiter='\t')

        local_f1score_arr, local_acc_arr, f1score_arr, acc_arr, mean_error, failure_rate = au_detection_eval(
            AUoccur_pred_prob, local_AUoccur_pred_prob, AUoccur_actual, pred_land, GT_land
        )

        # record result
        line1 = "Test model, train on {}, test on data: {}".format(config.task_fold, config.test_path_prefix)
        line2 = "F1 score of each au is: au1={}, au2={}, au4={}, au6={}, au9={}, au12={}, au25={}, au26={}".format(
                f1score_arr[0], f1score_arr[1], f1score_arr[2], f1score_arr[3],
                f1score_arr[4], f1score_arr[5], f1score_arr[6], f1score_arr[7])
        line3 = "Avarage F1 score is: avg={}".format(f1score_arr.mean())
        line4 = "Local F1 score of each au is: au1={}, au2={}, au4={}, au6={}, au9={}, au12={}, au25={}, au26={}".format(
                local_f1score_arr[0], local_f1score_arr[1], local_f1score_arr[2], local_f1score_arr[3],
                local_f1score_arr[4], local_f1score_arr[5], local_f1score_arr[6], local_f1score_arr[7])
        line5 = "Local Avarage F1 score is: avg={}".format(local_f1score_arr.mean())
        line6 = "Landmark mean error is: mean_error= {}".format(mean_error)
        line7 = "Acc of each au is: au1={}, au2={}, au4={}, au6={}, au9={}, au12={}, au25={}, au26={}".format(
            acc_arr[0], acc_arr[1], acc_arr[2], acc_arr[3],
            acc_arr[4], acc_arr[5], acc_arr[6], acc_arr[7])
        line8 = "Local acc is: local_acc_arr={}, Acc is: acc_arr={}".format(local_acc_arr.mean(),
                                                                            acc_arr.mean())

        res_file.write("===== Eval on Epoch {} =====".format(epoch))
        for line in [line1, line2, line3, line4, line5, line6, line7, line8]:
            res_file.write(line + '\n')
            logger.info(line+'\n')
        res_file.write("\n")

        # save chekpoints
        torch.save(net.state_dict(), config.task_log_prefix + '/epoch_{}.pth'.format(epoch))

    res_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Misc
    parser.add_argument('--training', type=bool, default=True, help='training or testing')
    parser.add_argument('--use_gpu', type=bool, default=True, help='default use gpu')
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--print_freq', type=int, default=100, help='interval of save checkpoints')

    # Training & Testing
    parser.add_argument('--train_batch_size', type=int, default=8, help='mini-batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='mini-batch size for evaluation')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=12, help='number of total epochs')
    parser.add_argument('--optimizer_type', type=str, default='SGD')
    parser.add_argument('--lr_type', type=str, default='step')
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.3, help='decay factor')
    parser.add_argument('--stepsize', type=int, default=2, help='epoch for decaying lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for SGD optimizer')
    parser.add_argument('--use_nesterov', type=str2bool, default=True)

    # Model configuration.
    parser.add_argument('--freeze_jaa', type=bool, default=True, help='freeze jaanet params')
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--map_size', type=int, default=44, help='size for attention maps')
    parser.add_argument('--au_num', type=int, default=8, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_land', type=float, default=0.5, help='weight for landmark detection loss')

    # Directories.
    parser.add_argument('--flip_reflect', type=str, default='./data/list/reflect_49.txt')
    parser.add_argument('--resume_model', type=str, default=None, help='resume from trained model')
    parser.add_argument('--task_fold', type=str, default='DISFA_combine_2_3')
    parser.add_argument('--task_log_prefix', type=str, default='./exps/jaav2_lstm/')
    parser.add_argument('--train_path_prefix', type=str, default='./data/list/DISFA_combine_2_3')
    parser.add_argument('--test_path_prefix', type=str, default='./data/list/DISFA_combine_2_3')
    parser.add_argument('--pretrain_prefix', type=str, default='./models/weights/DISFA_combine_2_3')
    config = parser.parse_args()

    config.task_log_prefix = config.task_log_prefix + config.task_fold
    if not os.path.exists(config.task_log_prefix):
        os.mkdir(config.task_log_prefix)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logger.add(config.task_log_prefix + '/{}.log'.format(cur_time))
    logger.info(config)
    logger.info('\n')

    main(config)