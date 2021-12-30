import argparse
import os
import torch.optim as optim
import torch.utils.data as util_data
from loguru import logger
import time

from models.single_aunet.aunet import AUNet
from utils.util import *
from data.tensor_list import TensorSingle
from utils import lr_schedule

optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}

torch.autograd.set_detect_anomaly(True)

def main(config):
    use_gpu = torch.cuda.is_available()
    config.use_gpu = use_gpu

    ## prepare data
    dsets = {}
    dset_loaders = {}

    dsets['train'] = TensorSingle(tensor_path=config.train_tensor_prefix)

    dset_loaders['train'] = util_data.DataLoader(dsets['train'], batch_size=config.train_batch_size,
                                                shuffle=False, num_workers=config.num_workers)

    dsets['test'] = TensorSingle(tensor_path=config.test_tensor_prefix)

    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config.eval_batch_size,
                                                shuffle=False, num_workers=config.num_workers)

    # set network modules
    net = AUNet(config)
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
            input, au = batch
            if use_gpu:
                input = input.cuda()
                au = au.long().cuda()
            else:
                au = [a.long() for a in au]

            # adjust
            optimizer = lr_scheduler(param_lr, optimizer, epoch, config.gamma, config.stepsize, config.init_lr)
            optimizer.zero_grad()
            # forward=
            loss_au_softmax, loss_au_dice = net(input, au)
            #loss_au_dice = config.train_batch_size * loss_au_dice
            total_loss = loss_au_softmax + loss_au_dice
            # backward
            total_loss.backward()
            optimizer.step()

            if i > 0 and i % config.print_freq == 0:
                line_l = "epoch={} || iter={} ||" \
                         " total_loss={:.4f} ||  loss_au_softmax={:.4f} || loss_au_dice={:.4f} || " \
                         "learning_rate={} \n".format(epoch, i,
                                                     total_loss.data.cpu().numpy(),
                                                     loss_au_softmax.data.cpu().numpy(),
                                                     loss_au_dice.data.cpu().numpy(),
                                                     optimizer.param_groups[0]['lr'])
                logger.info(line_l)

        # eval
        net.eval()
        net.training = False
        # each batch
        for i, batch in enumerate(dset_loaders['test']):
            input, au = batch
            if use_gpu:
                input = input.cuda()
                au = au.long().cuda()
            else:
                au = [a.long() for a in au]

            aus_output = net(input)
            if i == 0:
                all_output = aus_output.data.cpu().float()
                all_au = au.data.cpu().float()
            else:
                all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
                all_au = torch.cat((all_au, au.data.cpu().float()), 0)

        AUoccur_pred_prob = all_output.data.numpy()
        AUoccur_actual = all_au.data.numpy()

        # save AUoccur_pred_prob
        au_pred_file = config.task_log_prefix + '/Epoch{}_au_pred.txt'.format(epoch)
        np.savetxt(au_pred_file, AUoccur_pred_prob, fmt='%f', delimiter='\t')

        f1score_arr, acc_arr = au_detection_eval_v2(AUoccur_pred_prob, AUoccur_actual)

        # record result
        line1 = "Test model, train on {}, test on data: {}".format(config.task_fold, config.test_path_prefix)
        line2 = "F1 score of each au is: au1={}, au2={}, au4={}, au6={}, au9={}, au12={}, au25={}, au26={}".format(
                f1score_arr[0], f1score_arr[1], f1score_arr[2], f1score_arr[3],
                f1score_arr[4], f1score_arr[5], f1score_arr[6], f1score_arr[7])
        line3 = "Avarage F1 score is: avg={}".format(f1score_arr.mean())
        line4 = "Acc of each au is: au1={}, au2={}, au4={}, au6={}, au9={}, au12={}, au25={}, au26={}".format(
                acc_arr[0], acc_arr[1], acc_arr[2], acc_arr[3],
                acc_arr[4], acc_arr[5], acc_arr[6], acc_arr[7])
        line5 = "Avarage acc is: avg={}".format(acc_arr.mean())

        res_file.write("===== Eval on Epoch {} =====".format(epoch))
        for line in [line1, line2, line3, line4, line5]:
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
    parser.add_argument('--train_batch_size', type=int, default=32, help='mini-batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=80, help='mini-batch size for evaluation')
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
    # Model configuration
    parser.add_argument('--au_num', type=int, default=8, help='number of AUs')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    # Directories.
    parser.add_argument('--pretrain_prefix', type=str, default='../weights/DISFA_combine_1_2')
    parser.add_argument('--resume_model', type=str, default=None, help='resume from trained model')
    parser.add_argument('--task_log_prefix', type=str, default='./exps/tnet_multi/')
    parser.add_argument('--task_fold', type=str, default='DISFA_combine_2_3')
    parser.add_argument('--train_path_prefix', type=str, default='./data/list/DISFA_combine_2_3')
    parser.add_argument('--train_tensor_prefix', type=str, default='./data/tensor_12000/DISFA_combine_2_3/train')
    parser.add_argument('--test_path_prefix', type=str, default='./data/list/DISFA_combine_2_3')
    parser.add_argument('--test_tensor_prefix', type=str, default='./data/tensor_12000/DISFA_combine_2_3/test')

    config = parser.parse_args()

    if not os.path.exists(config.task_log_prefix):
        os.mkdir(config.task_log_prefix)
    config.task_log_prefix = config.task_log_prefix + config.task_fold
    if not os.path.exists(config.task_log_prefix):
        os.mkdir(config.task_log_prefix)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logger.add(config.task_log_prefix + '/{}.log'.format(cur_time))
    logger.info(config)
    logger.info('\n')

    main(config)