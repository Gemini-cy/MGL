import torch
from torch import nn, optim
import timeit
import numpy as np
import os
from tensorboardX import SummaryWriter
from collections import OrderedDict

import argparse
import torch.distributed as dist
import time

# 加载多视角融合模型
from net.MCL_backbone.mcl_stgcn import MCL_STGCN
from net.MCL_backbone.mcl_agcn import MCL_AGCN
from net.MCL_backbone.mcl_shiftgcn import MCL_SHIFTGCN
from net.MCL_backbone.mcl_msg3d import MCL_MSG3D
from net.MCL_backbone.mcl_ctrgcn import MCL_CTRGCN

from feeder.feeder_fusion import Feeder, MultiDataset

# 初始化ST-GCN某些网络层的初始参数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("Conv2d") != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 计算Top1、Top5准确率
def show_topk(k, result, label):
    rank = result.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


# 多个进程之间的准确率和loss计算
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()   # 总进程数
    return rt


def processor(modelName, optimizer_name, lr, device_list, debug,
            train_batch_size_list, test_batch_size_list, num_epoch, save_dir_root,
              save_epoch):
    # 加载模型
    if modelName == 'MCL_STGCN':
        model = MCL_STGCN(in_channels=3, num_class=60, device_id=device_list[0], edge_importance_weighting=True,
                          graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
        model.apply(weights_init)
    elif modelName == 'MCL_AGCN':
        model = MCL_AGCN(in_channels=3, num_class=60, device_id=device_list[0], edge_importance_weighting=True,
                         graph_args_agcn={'labeling_mode': 'spatial'},
                         graph_args_stgcn={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'MCL_SHIFTGCN':
        model = MCL_SHIFTGCN(in_channels=3, num_class=60, device_id=device_list[0], edge_importance_weighting=True,
                         graph_args_shiftgcn={'labeling_mode': 'spatial'},
                         graph_args_stgcn={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'MCL_MSG3D':
        model = MCL_MSG3D(in_channels=3, num_class=60, num_point=25, num_person=2,
                          device_id=device_list[0], num_gcn_scales=13, num_g3d_scales=6,
                          graph_args_msg3d={"net.utils.msg3d.graph.AdjMatrixGraph"},
                          graph_args_stgcn={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                          edge_importance_weighting=True)
    elif modelName == 'MCL_CTRGCN':
        model = MCL_CTRGCN(in_channels=3, num_class=60, device_id=device_list[0], edge_importance_weighting=True,
                         graph_args_ctrgcn={'labeling_mode': 'spatial'},
                         graph_args_stgcn={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    else:
        print("We only implemented MCL_STGCN, MCL_AGCN models.")
        raise NotImplementedError

    # 损失函数及优化器
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1, last_epoch=-1)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=0.0001)
    else:
        print("We only use SGD optimizer and Adam optimizer.")
        raise ValueError()

    # 计算模型参数
    if args.local_rank==1:
        # 主机卡 -- 3号
        train_batch_size = train_batch_size_list[0]
        test_batch_size = test_batch_size_list[0]
    elif args.local_rank==0:
        # 2号显卡
        train_batch_size = train_batch_size_list[1]
        test_batch_size = test_batch_size_list[1]
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # 将模型传入设备
    # device = torch.device("cuda", args.local_rank)
    # model.to(device)
    # device = torch.device("cuda:"+str(device_list[0]) if torch.cuda.is_available() else "cpu")
    # print(device)
    # if len(device_list) > 1:
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
        # model = nn.DataParallel(model, device_ids=device_list).cuda()
    # model.to(device)

    # 将交叉熵损失函数传入设备
    # criterion.to(device)
    criterion.cuda(args.local_rank)

    # 加载数据
    data_loader = dict()
    # 三个视角的训练集路径
    train_view1_data = "data/ntu60_rotation45/xsub/train/train_view1_data.npy"
    train_view1_label = "data/ntu60_rotation45/xsub/train/train_view1_label.pkl"
    train_view2_data = "data/ntu60_rotation45/xsub/train/train_view2_data.npy"
    train_view2_label = "data/ntu60_rotation45/xsub/train/train_view2_label.pkl"
    train_view3_data = "data/ntu60_rotation45/xsub/train/train_view3_data.npy"
    train_view3_label = "data/ntu60_rotation45/xsub/train/train_view3_label.pkl"
    train_view1 = Feeder(data_path=train_view1_data, label_path=train_view1_label, debug=debug)
    train_view2 = Feeder(data_path=train_view2_data, label_path=train_view2_label, debug=debug)
    train_view3 = Feeder(data_path=train_view3_data, label_path=train_view3_label, debug=debug)
    multi_train = MultiDataset(train_view1, train_view2, train_view3)
    # 每个进程一个sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(multi_train)
    data_loader["train"] = torch.utils.data.DataLoader(
        dataset=multi_train,
        batch_size=train_batch_size,
        # shuffle=True,
        num_workers=0*len(device_list),
        sampler=train_sampler
    )
    # 三个视角的测试集路径
    test_view1_data = "data/ntu60_rotation45/xsub/val/val_view1_data.npy"
    test_view1_label = "data/ntu60_rotation45/xsub/val/val_view1_label.pkl"
    test_view2_data = "data/ntu60_rotation45/xsub/val/val_view2_data.npy"
    test_view2_label = "data/ntu60_rotation45/xsub/val/val_view2_label.pkl"
    test_view3_data = "data/ntu60_rotation45/xsub/val/val_view3_data.npy"
    test_view3_label = "data/ntu60_rotation45/xsub/val/val_view3_label.pkl"
    test_view1 = Feeder(data_path=test_view1_data, label_path=test_view1_label, debug=debug)
    test_view2 = Feeder(data_path=test_view2_data, label_path=test_view2_label, debug=debug)
    test_view3 = Feeder(data_path=test_view3_data, label_path=test_view3_label, debug=debug)
    multi_test = MultiDataset(test_view1, test_view2, test_view3)
    # 每个进程一个sampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(multi_test)
    data_loader["test"] = torch.utils.data.DataLoader(
        dataset=multi_test,
        batch_size=test_batch_size,
        # shuffle=True,
        num_workers=0*len(device_list),
        sampler=test_sampler
    )

    # 日志保存路径
    log_dir = os.path.join(save_dir_root, 'run', 'run_' + modelName)
    writer = SummaryWriter(log_dir=log_dir)

    # 开始训练测试
    for epoch in range(num_epoch):
        for phase in ["train", "test"]:
            # 训练阶段
            if phase == "train":
                start_time = timeit.default_timer()
                train_sampler.set_epoch(epoch)   # 打乱训练集
                model.train()
                labels_len = 0
                # 多个进程间的loss和acc计算
                multi_train_loss = 0
                multi_train_acc1 = 0
                multi_train_acc5 = 0
                for batch_index, batch_data in enumerate(data_loader["train"]):
                    # print("start: ", torch.cuda.memory_allocated()/1024/1024)
                    view1, view2, view3 = batch_data[0], batch_data[1], batch_data[2]
                    # view1_data = view1[0].float().to(device)
                    view1_data = view1[0].float().cuda(args.local_rank)
                    view1_label = view1[1].float().cuda(args.local_rank)
                    view2_data = view2[0].float().cuda(args.local_rank)
                    view2_label = view2[1].float().cuda(args.local_rank)
                    view3_data = view3[0].float().cuda(args.local_rank)
                    view3_label = view3[1].float().cuda(args.local_rank)
                    labels_len += len(view1_label)
                    # print(batch_index, view1_data.size(),view2_data.size(),view3_data.size())
                    output1, output2 = model(view1_data, view2_data, view3_data)
                    # print("end: ", torch.cuda.memory_allocated() / 1024 / 1024)
                    # stage1 -- loss计算
                    stage1_loss1 = criterion(output1[0], view1_label.long())
                    stage1_loss2 = criterion(output1[1], view2_label.long())
                    stage1_loss3 = criterion(output1[2], view3_label.long())
                    # stage2 -- loss计算
                    stage2_loss = criterion(output2, view1_label.long())
                    # 所有loss计算
                    loss_sum = stage1_loss1 + stage1_loss2 + stage1_loss3 + stage2_loss

                    # 把多个进程的loss和准确率加起来
                    acc_top1 = show_topk(1, output2.data.cpu().numpy(), view1_label.data.cpu().numpy())
                    acc_top5 = show_topk(5, output2.data.cpu().numpy(), view1_label.data.cpu().numpy())
                    # 将每个显卡上的进程里的loss，top1，top5重新计算
                    reduced_loss = reduce_tensor(loss_sum).item()
                    reduced_acc_top1 = reduce_tensor(torch.tensor(acc_top1).cuda(args.local_rank)).item()
                    reduced_acc_top5 = reduce_tensor(torch.tensor(acc_top5).cuda(args.local_rank)).item()
                    multi_train_loss += reduced_loss * len(view1_label)
                    multi_train_acc1 += reduced_acc_top1 * len(view1_label)
                    multi_train_acc5 += reduced_acc_top5 * len(view1_label)

                    # backward
                    optimizer.zero_grad()
                    loss_sum.backward()
                    optimizer.step()
                    # for devide_id in device_list:
                    #     with torch.cuda.device('cuda:'+str(devide_id)):
                    #         torch.cuda.empty_cache()
                    # torch.cuda.empty_cache()

                # 更改学习率
                if optimizer_name == "SGD":
                    scheduler.step()

                # 计算train过程的Top1，Top5，loss值
                train_loss = multi_train_loss / labels_len
                train_acc1 = multi_train_acc1 / labels_len
                train_acc5 = multi_train_acc5 / labels_len
                # print(train_loss, train_acc1, train_acc5)
                # print(args.local_rank)
                # SummaryWriter写入数据操作
                if args.local_rank == 0:
                    writer.add_scalar('data/train_loss_epoch', train_loss, epoch)
                    writer.add_scalar('data/train_acc_top1_epoch', train_acc1, epoch)
                    writer.add_scalar('data/train_acc_top5_epoch', train_acc5, epoch)
                    stop_time = timeit.default_timer()
                    print("[train] Epoch: {}/{} lr:{} Loss:{} Top1_acc:{} Top5_acc:{}".format(epoch+1,num_epoch,optimizer.state_dict()['param_groups'][0]['lr'],train_loss,train_acc1, train_acc5))
                    print("Execution time: " + str((stop_time - start_time)))

            # 储存模型

            if (epoch+1) % save_epoch == 0 and phase == "train" and args.local_rank == 0:
                model_save_dir = os.path.join(save_dir_root, "model", modelName, "Epoch{}_".format(epoch+1)+modelName+".pt")
                state_dict = model.state_dict()
                torch.save(state_dict, model_save_dir)
                print("Save model at {}".format(model_save_dir))

            # test阶段
            if phase=="test":
                model.eval()
                start_time = timeit.default_timer()
                labels_len = 0
                # 多个进程间的loss和acc计算
                multi_test_loss = 0
                multi_test_acc1 = 0
                multi_test_acc5 = 0
                for batch_index, batch_data in enumerate(data_loader["test"]):
                    view1, view2, view3 = batch_data[0], batch_data[1], batch_data[2]
                    view1_data = view1[0].float().cuda(args.local_rank)
                    view1_label = view1[1].float().cuda(args.local_rank)
                    view2_data = view2[0].float().cuda(args.local_rank)
                    view2_label = view2[1].float().cuda(args.local_rank)
                    view3_data = view3[0].float().cuda(args.local_rank)
                    view3_label = view3[1].float().cuda(args.local_rank)
                    labels_len += len(view1_label)
                    with torch.no_grad():
                        output1, output2 = model(view1_data, view2_data, view3_data)
                    # stage1 -- loss计算
                    stage1_loss1 = criterion(output1[0], view1_label.long())
                    stage1_loss2 = criterion(output1[1], view2_label.long())
                    stage1_loss3 = criterion(output1[2], view3_label.long())
                    # stage2 -- loss计算
                    stage2_loss = criterion(output2, view1_label.long())
                    # 所有loss计算
                    loss_sum = stage1_loss1 + stage1_loss2 + stage1_loss3 + stage2_loss

                    # 把多个进程的loss和准确率加起来
                    acc_top1 = show_topk(1, output2.data.cpu().numpy(), view1_label.data.cpu().numpy())
                    acc_top5 = show_topk(5, output2.data.cpu().numpy(), view1_label.data.cpu().numpy())
                    # 将每个显卡上的进程里的loss，top1，top5重新计算
                    reduced_loss = reduce_tensor(loss_sum).item()
                    reduced_acc_top1 = reduce_tensor(torch.tensor(acc_top1).cuda(args.local_rank)).item()
                    reduced_acc_top5 = reduce_tensor(torch.tensor(acc_top5).cuda(args.local_rank)).item()
                    multi_test_loss += reduced_loss * len(view1_label)
                    multi_test_acc1 += reduced_acc_top1 * len(view1_label)
                    multi_test_acc5 += reduced_acc_top5 * len(view1_label)

                # 计算test过程的Top1，Top5，loss值
                test_loss = multi_test_loss / labels_len
                test_acc1 = multi_test_acc1 / labels_len
                test_acc5 = multi_test_acc5 / labels_len
                # SummaryWriter写入数据操作
                if args.local_rank == 0:
                    writer.add_scalar('data/test_loss_epoch', test_loss, epoch)
                    writer.add_scalar('data/test_acc_top1_epoch', test_acc1, epoch)
                    writer.add_scalar('data/test_acc_top5_epoch', test_acc5, epoch)
                    stop_time = timeit.default_timer()
                    print("[test] Epoch: {}/{} Loss:{} Top1_acc:{} Top5_acc:{}".format(epoch+1,num_epoch,test_loss, test_acc1, test_acc5))
                    print("Execution time: " + str((stop_time - start_time)) + "\n")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=1, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    modelName = 'MCL_MSG3D'
    optimizer_name = 'SGD'   # ['SGD',lr=0.1]   ['Adam', lr=0.001]
    lr = 0.05
    device = [0, 1]   # 主楼顺序：device0-主楼1  device1-主楼3  device2-主楼0  device3-主楼2
    debug = False
    train_batch_size = [6,6]
    test_batch_size = [64,64]
    num_epoch = 80
    # 储存路径 -- 当前根目录
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    # save_epoch表示训练每多少轮存一次模型
    save_epoch = 1
    processor(modelName=modelName, optimizer_name=optimizer_name, lr=lr,
              device_list=device, debug=debug, train_batch_size_list=train_batch_size,
              test_batch_size_list=test_batch_size, num_epoch=num_epoch,
              save_dir_root=save_dir_root, save_epoch=save_epoch)

# 命令行输入
# CUDA_VISIBLE_DEVICES=3,1 python -m torch.distributed.launch --nproc_per_node=2 train_backbone_distributed.py


# 默认参数设置(ST-GCN,AGCN,ShiftGCN)： lr=0.1, weight_decay=0.0001,step:[30,50],epoch:80
# MS-G3D参数设置：lr=0.05， weight_decay=0.0005, step:[30,40],epoch:50
# CTR-GCN参数设置：lr=0.1, weight_decay=0.004,step:[35,55],epoch:65

