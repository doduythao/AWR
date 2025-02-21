JOINT = {
    'nyu': 14,
    'icvl': 16,
    'msra': 21,
    'hands17': 21
}
STEP = {
    'nyu': 30,
    'icvl': 10,
    'msra': 10,
    'hands17': 25
}
EPOCH = {
    'nyu': 40,
    'icvl': 40,
    'msra': 25,
    'hands17': 15
}
class Config(object):
    gpu_id = 0
    exp_id = "hands17_hourglass_e10"
    log_id = "dense"

    data_dir = './data'
    dataset = 'hands17'
    output_dir = './output/'
    load_model = './results/hands17_best_e3.pth'

    jt_num = JOINT[dataset]
    cube = [300, 300, 300]
    augment_para = [10, 0.1, 120]

    net = 'hourglass_1' # 'resnet_18'
    downsample = 2 # [1,2,4]
    img_size = 128
    batch_size = 32
    num_workers = 8
    max_epoch = EPOCH[dataset]
    loss_type = 'MyL1Loss'
    dense_weight = 1.
    coord_weight = 0.5
    angle_w = 0
    ratio_w = 0
    kernel_size = 0.4 # 0.4 for hourglass and 1 for resnet
    lr = 1e-3
    optimizer = 'sgd'
    scheduler = 'step'
    step = STEP[dataset]
    weight_decay = 0
    print_freq = 100
    vis_freq = 1

opt = Config()

