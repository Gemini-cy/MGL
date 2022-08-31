import torch
from torch import nn, optim
import timeit
import numpy as np
import os
from tensorboardX import SummaryWriter
from collections import OrderedDict

# 加载多视角融合模型
from net.MultiView import CF_model
from net.MultiView import CS_model
from net.MultiView import CNNS_model
from net.MultiView import MP_model
from net.MultiView import AP_model
from net.MultiView import LSTM_model
from net.MultiView import MCL_model

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


def processor(modelName, optimizer_name, lr, device_list, debug,
            train_batch_size, test_batch_size, num_epoch, save_dir_root,
              save_epoch):
    # 加载模型
    if modelName == 'CF_model':
        model = CF_model.ConcatenateFeature(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'CS_model':
        model = CS_model.ConcatenateScore(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'CNNS_model':
        model = CNNS_model.ConvolutionalScore(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'MP_model':
        model = MP_model.MaxPoolingFeature(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'AP_model':
        model = AP_model.AveragePoolingFeature(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'LSTM_model':
        model = LSTM_model.LstmFeature(in_channels=3,num_class=60,edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    elif modelName == 'MCL_model':
        model = MCL_model.MCL(in_channels=3,num_class=60,device_id=device_list[0],edge_importance_weighting=True,
                    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
    else:
        print("We only implemented CF, CS, CNNS, MP, AP, LSTM and MCL models.")
        raise NotImplementedError
    # 初始化模型参数
    model.apply(weights_init)

    # 损失函数及优化器
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1, last_epoch=-1)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=0.0001)
    else:
        print("We only use SGD optimizer and Adam optimizer.")
        raise ValueError()

    # 计算模型参数
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # 将模型传入设备
    device = torch.device("cuda:"+str(device_list[0]) if torch.cuda.is_available() else "cpu")
    if len(device_list) > 1:
        model = nn.DataParallel(model, device_ids=device_list)
    model.to(device)

    # 将交叉熵损失函数传入设备
    criterion.to(device)

    # 加载数据
    data_loader = dict()
    # 三个视角的训练集路径
    train_view1_data = "data/spatial_temporal_occ/ntu60_rotation45/xsub/train/train_view1_data.npy"
    train_view1_label = "data/spatial_temporal_occ/ntu60_rotation45/xsub/train/train_view1_label.pkl"
    train_view2_data = "data/spatial_temporal_occ/ntu60_rotation45/xsub/train/train_view2_data.npy"
    train_view2_label = "data/spatial_temporal_occ/ntu60_rotation45/xsub/train/train_view2_label.pkl"
    train_view3_data = "data/spatial_temporal_occ/ntu60_rotation45/xsub/train/train_view3_data.npy"
    train_view3_label = "data/spatial_temporal_occ/ntu60_rotation45/xsub/train/train_view3_label.pkl"
    train_view1 = Feeder(data_path=train_view1_data, label_path=train_view1_label, debug=debug)
    train_view2 = Feeder(data_path=train_view2_data, label_path=train_view2_label, debug=debug)
    train_view3 = Feeder(data_path=train_view3_data, label_path=train_view3_label, debug=debug)
    multi_train = MultiDataset(train_view1, train_view2, train_view3)
    data_loader["train"] = torch.utils.data.DataLoader(
        dataset=multi_train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0*len(device_list)
    )
    # 三个视角的测试集路径
    test_view1_data = "data/spatial_temporal_occ/ntu60_rotation45/xsub/val/val_view1_data.npy"
    test_view1_label = "data/spatial_temporal_occ/ntu60_rotation45/xsub/val/val_view1_label.pkl"
    test_view2_data = "data/spatial_temporal_occ/ntu60_rotation45/xsub/val/val_view2_data.npy"
    test_view2_label = "data/spatial_temporal_occ/ntu60_rotation45/xsub/val/val_view2_label.pkl"
    test_view3_data = "data/spatial_temporal_occ/ntu60_rotation45/xsub/val/val_view3_data.npy"
    test_view3_label = "data/spatial_temporal_occ/ntu60_rotation45/xsub/val/val_view3_label.pkl"
    test_view1 = Feeder(data_path=test_view1_data, label_path=test_view1_label, debug=debug)
    test_view2 = Feeder(data_path=test_view2_data, label_path=test_view2_label, debug=debug)
    test_view3 = Feeder(data_path=test_view3_data, label_path=test_view3_label, debug=debug)
    multi_test = MultiDataset(test_view1, test_view2, test_view3)
    data_loader["test"] = torch.utils.data.DataLoader(
        dataset=multi_test,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=0*len(device_list)
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
                model.train()
                result_frag = []
                label_frag = []
                loss_value = []
                labels_len = 0
                for batch_index, batch_data in enumerate(data_loader["train"]):
                    # print("start: ", torch.cuda.memory_allocated()/1024/1024)
                    view1, view2, view3 = batch_data[0], batch_data[1], batch_data[2]
                    view1_data = view1[0].float().to(device)
                    view1_label = view1[1].float().to(device)
                    view2_data = view2[0].float().to(device)
                    view2_label = view2[1].float().to(device)
                    view3_data = view3[0].float().to(device)
                    view3_label = view3[1].float().to(device)
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

                    # 记录train过程中的结果
                    result_frag.append(output2.data.cpu().numpy())
                    label_frag.append(view1_label.data.cpu().numpy())

                    # backward
                    optimizer.zero_grad()
                    loss_sum.backward()
                    optimizer.step()
                    # for devide_id in device_list:
                    #     with torch.cuda.device('cuda:'+str(devide_id)):
                    #         torch.cuda.empty_cache()
                    # torch.cuda.empty_cache()

                    # train过程loss值统计
                    loss_value.append(loss_sum.data.item())

                # 更改学习率
                if optimizer_name == "SGD":
                    scheduler.step()

                # 计算train过程的Top1，Top5，loss值
                result = np.concatenate(result_frag)
                label = np.concatenate(label_frag)
                acc_top1 = show_topk(1, result, label)
                acc_top5 = show_topk(5, result, label)
                train_loss = np.mean(loss_value)
                # SummaryWriter写入数据操作
                writer.add_scalar('data/train_loss_epoch', train_loss, epoch)
                writer.add_scalar('data/train_acc_top1_epoch', acc_top1, epoch)
                writer.add_scalar('data/train_acc_top5_epoch', acc_top5, epoch)
                stop_time = timeit.default_timer()
                print("[train] Epoch: {}/{} lr:{} Loss:{} Top1_acc:{} Top5_acc:{}".format(epoch+1,num_epoch,optimizer.state_dict()['param_groups'][0]['lr'],train_loss, acc_top1, acc_top5))
                print("Execution time: " + str((stop_time - start_time)))

            # 储存模型
            if (epoch+1) % save_epoch == 0 and phase == "train":
                model_save_dir = os.path.join(save_dir_root, "model", modelName, "Epoch{}_".format(epoch+1)+modelName+".pt")
                state_dict = model.state_dict()
                torch.save(state_dict, model_save_dir)
                print("Save model at {}".format(model_save_dir))

            # test阶段
            if phase=="test":
                model.eval()
                start_time = timeit.default_timer()
                result_frag = []
                label_frag = []
                loss_value = []
                labels_len = 0
                for batch_index, batch_data in enumerate(data_loader["test"]):
                    view1, view2, view3 = batch_data[0], batch_data[1], batch_data[2]
                    view1_data = view1[0].float().to(device)
                    view1_label = view1[1].float().to(device)
                    view2_data = view2[0].float().to(device)
                    view2_label = view2[1].float().to(device)
                    view3_data = view3[0].float().to(device)
                    view3_label = view3[1].float().to(device)
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

                    # 记录test过程中的结果
                    result_frag.append(output2.data.cpu().numpy())
                    label_frag.append(view1_label.data.cpu().numpy())
                    # test过程loss值统计
                    loss_value.append(loss_sum.item())

                # 计算test过程的Top1，Top5，loss值
                result = np.concatenate(result_frag)
                label = np.concatenate(label_frag)
                acc_top1 = show_topk(1, result, label)
                acc_top5 = show_topk(5, result, label)
                test_loss = np.mean(loss_value)
                # SummaryWriter写入数据操作
                writer.add_scalar('data/test_loss_epoch', test_loss, epoch)
                writer.add_scalar('data/test_acc_top1_epoch', acc_top1, epoch)
                writer.add_scalar('data/test_acc_top5_epoch', acc_top5, epoch)
                stop_time = timeit.default_timer()
                print("[test] Epoch: {}/{} Loss:{} Top1_acc:{} Top5_acc:{}".format(epoch+1,num_epoch,test_loss, acc_top1, acc_top5))
                print("Execution time: " + str((stop_time - start_time)) + "\n")
    writer.close()


if __name__ == "__main__":
    modelName = 'MP_model'  # 'CF_model'
    optimizer_name = 'SGD'   # ['SGD',lr=0.1]   ['Adam', lr=0.001]
    lr = 0.1
    device = [0,1,2]
    debug = False
    train_batch_size = 32
    test_batch_size = 64
    num_epoch = 80
    # 储存路径 -- 当前根目录
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    # save_epoch表示训练每多少轮存一次模型
    save_epoch = 1
    processor(modelName=modelName, optimizer_name=optimizer_name, lr=lr,
              device_list=device, debug=debug, train_batch_size=train_batch_size,
              test_batch_size=test_batch_size, num_epoch=num_epoch,
              save_dir_root=save_dir_root, save_epoch=save_epoch)


# CF_model lr=0.1 SGD train_batch_size=32 [10,30]
# MP_model lr=0.1 SGD train_batch_size=32 [10,30]






