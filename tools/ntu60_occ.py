import os
import sys
import pickle
import math
import random
import numpy as np
from numpy.lib.format import open_memmap

training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30
camera_view = 3

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


"""读取skeleton文件"""
def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


"""对单一样本进行旋转操作"""
def rotation_xyz(data, angle):
    # 旋转数据操作
    # ntu沿y轴
    rotation_matrix_y = np.array([[math.cos(angle), 0, -math.sin(angle)],
                                  [0, 1, 0],
                                  [math.sin(angle), 0, math.cos(angle)]])
    c,j,m = data.shape
    data = data.reshape((c,j*m))
    data = np.dot(rotation_matrix_y,data)
    data = data.reshape((c,j,m))
    return data


"""对旋转后的样本骨骼范围进行测算"""
def skel_boundary(data):
    maxX, minX = data[0, :, 0].max(), data[0, :, 0].min()
    maxY, minY = data[1, :, 0].max(), data[1, :, 0].min()
    return maxX, minX, maxY, minY


"""判断是否满足四边形条件"""
def judge_quadrilateral(coordinate, occ1, occ2):
    world_coordinate = np.array(coordinate)
    # 遮挡操作
    occ1_maxX, occ1_minX, occ1_maxY, occ1_minY = occ1
    occ2_maxX, occ2_minX, occ2_maxY, occ2_minY = occ2
    occ1_vertex_a = np.array([occ1_minX, occ1_maxY, 2.6])
    occ1_vertex_b = np.array([occ1_maxX, occ1_maxY, 2.6])
    occ1_vertex_c = np.array([occ1_maxX, occ1_minY, 2.6])
    occ1_vertex_d = np.array([occ1_minX, occ1_minY, 2.6])
    z1 = np.dot((occ1_vertex_b - occ1_vertex_a), (world_coordinate - occ1_vertex_a))
    z2 = np.dot((occ1_vertex_c - occ1_vertex_b), (world_coordinate - occ1_vertex_b))
    z3 = np.dot((occ1_vertex_d - occ1_vertex_c), (world_coordinate - occ1_vertex_c))
    z4 = np.dot((occ1_vertex_a - occ1_vertex_d), (world_coordinate - occ1_vertex_d))

    occ2_vertex_a = np.array([occ2_minX, occ2_maxY, 2.6])
    occ2_vertex_b = np.array([occ2_maxX, occ2_maxY, 2.6])
    occ2_vertex_c = np.array([occ2_maxX, occ2_minY, 2.6])
    occ2_vertex_d = np.array([occ2_minX, occ2_minY, 2.6])
    z5 = np.dot((occ2_vertex_b - occ2_vertex_a), (world_coordinate - occ2_vertex_a))
    z6 = np.dot((occ2_vertex_c - occ2_vertex_b), (world_coordinate - occ2_vertex_b))
    z7 = np.dot((occ2_vertex_d - occ2_vertex_c), (world_coordinate - occ2_vertex_c))
    z8 = np.dot((occ2_vertex_a - occ2_vertex_d), (world_coordinate - occ2_vertex_d))
    is_positive1 = (z1 * z2 > 0) and (z3 * z4 > 0) and (z1 * z3 > 0)
    is_positive2 = (z5 * z6 > 0) and (z7 * z8 > 0) and (z5 * z7 > 0)
    if is_positive1 or is_positive2:
        image_coordinate = np.array([0, 0, 0])
    else:
        image_coordinate = world_coordinate
    return image_coordinate


"""遮挡操作"""
def occlusion(data, angles):
    # 沿y轴旋转
    data_left = rotation_xyz(data, angles[0])
    data_right = rotation_xyz(data, angles[1])
    # 寻找三个视角骨骼的边界范围
    maxX1, minX1, maxY1, minY1 = skel_boundary(data_left)
    maxX2, minX2, maxY2, minY2 = skel_boundary(data)
    maxX3, minX3, maxY3, minY3 = skel_boundary(data_right)
    # 左边视角取骨骼右半部分1/2,中间视角取骨骼左半部分1/3，右半部分1/3,右边视角取骨骼左半部分1/2
    occ1_minX, occ1_maxX = maxX1 - (maxX1 - minX1) / 2, minX2 + (maxX2 - minX2) / 3
    occ1_minY, occ1_maxY = min(minY1, minY2), max(maxY1, maxY2)
    occ2_minX, occ2_maxX = maxX2 - (maxX2 - minX2) / 3, minX3 + (maxX3 - minX3) / 2
    occ2_minY, occ2_maxY = min(minY2, minY3), max(maxY2, maxY3)
    occ1 = (occ1_maxX, occ1_minX, occ1_maxY, occ1_minY)
    occ2 = (occ2_maxX, occ2_minX, occ2_maxY, occ2_minY)
    # 将遮挡范围内的骨骼点置为0
    for m in range(max_body):
        for j in range(num_joint):
            data_left[:,j,m] = judge_quadrilateral(data_left[:,j,m], occ1, occ2)
            data[:, j, m] = judge_quadrilateral(data[:, j, m], occ1, occ2)
            data_right[:, j, m] = judge_quadrilateral(data_right[:, j, m], occ1, occ2)
    return data_left, data, data_right


"""时间遮挡"""
def time_occ(seed_num):
    random.seed(seed_num[0])
    time_num1 = random.randint(0, 99)
    random.seed(seed_num[1])
    time_num2 = random.randint(100, 199)
    random.seed(seed_num[2])
    time_num3 = random.randint(200, 299)
    return time_num1, time_num2, time_num3

""""读取每个样本的每帧"""
def data_operation(file, angles, seed_num):
    seq_info = read_skeleton(file)
    # 设定时间遮挡的随机数种子
    random.seed(seed_num)
    seed_num_list = [random.randint(0, 300) for _ in range(9)]
    data = np.zeros((3, max_frame, num_joint, max_body))
    data_left = np.zeros((3, max_frame, num_joint, max_body))
    data_right = np.zeros((3, max_frame, num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
        # 进行完整旋转自适应空间遮挡和时间遮挡操作
        frame_data_left, frame_data, frame_data_right = occlusion(data[:,n,:,:],angles)

        data[:,n,:,:] = frame_data
        if n in time_occ(seed_num_list[0:3]):
            data[:, n, :, :] = 0   # 时间遮挡

        data_left[:,n,:,:] = frame_data_left
        if n in time_occ(seed_num_list[3:6]):
            data_left[:, n, :, :] = 0   # 时间遮挡

        data_right[:,n,:,:] = frame_data_right
        if n in time_occ(seed_num_list[6:9]):
            data_right[:, n, :, :] = 0   # 时间遮挡

    return data, data_left, data_right



def gendata_xsub(data_path, out_path, angles, ignored_sample_path=None, part='eval', benchmark='xsub'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        print("error!!!!!!")
        ignored_samples = []

    # 储存样本名字
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        # 判断当前样本是属于训练集还是测试集
        istraining = (subject_id in training_subjects)
        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()
        # 对当前样本进行分类
        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    # 旋转遮挡操作
    for view in range(camera_view):
        with open('{}/{}_view{}_label.pkl'.format(out_path, part, view+1), 'wb') as f:
            pickle.dump((sample_name, list(sample_label)), f)
        # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp_view1 = open_memmap('{}/{}_view1_data.npy'.format(out_path, part),dtype='float32',mode='w+',
                     shape=(len(sample_label), 3, max_frame, num_joint, max_body))
    fp_view2 = open_memmap('{}/{}_view2_data.npy'.format(out_path, part), dtype='float32', mode='w+',
                           shape=(len(sample_label), 3, max_frame, num_joint, max_body))
    fp_view3 = open_memmap('{}/{}_view3_data.npy'.format(out_path, part), dtype='float32', mode='w+',
                           shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    # 设定时间遮挡的随机数种子
    seed = 615
    random.seed(seed)
    seed_num = [random.randrange(i, 1500000+i, 3) for i in range(len(sample_name))]
    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data, data_left, data_right = data_operation(os.path.join(data_path, s), angles, seed_num[i])
        fp_view1[i, :, 0:data_left.shape[1], :, :] = data_left
        fp_view2[i, :, 0:data.shape[1], :, :] = data
        fp_view3[i, :, 0:data_right.shape[1], :, :] = data_right
    end_toolbar()








if __name__ == '__main__':
    data_path = "../../NTU-RGB-D"
    out_path = "../data/ntu60_rotation45"
    ignored_sample_path = "../resource/ntu60/samples_with_missing_skeletons.txt"
    benchmark = ["xsub"]
    part = ['train', 'val']
    angles = [-math.pi/4, math.pi/4]

    for b in benchmark:
        for p in part:
            out_path = os.path.join(out_path, b, p)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata_xsub(data_path, out_path, angles, ignored_sample_path, benchmark=b, part=p)

