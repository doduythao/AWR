from PIL import Image
import numpy as np
from os import path

from dataloader.loader import Loader
from util.util import uvd2xyz, xyz2uvd


class Hands17(Loader):
    def __init__(
        self,
        root,
        phase,
        val=False,
        img_size=128,
        aug_para=[10, 0.1, 180],
        cube=[300, 300, 300],
        jt_num=21,
    ):
        super(Hands17, self).__init__(root, phase, img_size, "HANDS17")
        self.name = "HANDS17"
        self.root = root
        self.phase = phase
        self.val = val

        self.paras = (475.065948, 475.065857, 315.944855, 245.287079)  # camera info!
        self.cube = np.asarray(cube)
        self.dsize = np.asarray([img_size, img_size])
        self.img_size = img_size

        self.jt_num = jt_num
        self.aug_para = aug_para
        self.flip = 1
        self.data = self.make_dataset()

        print("loading dataset, containing %d images." % len(self.data))

    def __getitem__(self, index):
        img = self.img_reader(self.data[index][0])
        jt_xyz = self.data[index][2].copy()

        cube = self.cube

        center_xyz = self.data[index][3].copy()
        center_uvd = xyz2uvd(center_xyz, self.paras, self.flip)

        jt_xyz -= center_xyz
        img, M = self.crop(img, center_uvd, cube, self.dsize)

        if self.phase == "train" and self.val == False:
            aug_op, trans, scale, rot = self.random_aug(*self.aug_para)
            img, jt_xyz, cube, center_uvd, M = self.augment(
                img, jt_xyz, center_uvd, cube, M, aug_op, trans, scale, rot
            )
            center_xyz = uvd2xyz(center_uvd, self.paras, self.flip)
        else:
            img = self.normalize(img.max(), img, center_xyz, cube)

        jt_uvd = self.transform_jt_uvd(
            xyz2uvd(jt_xyz + center_xyz, self.paras, self.flip), M
        )
        jt_uvd[:, :2] = jt_uvd[:, :2] / (self.img_size / 2.0) - 1
        jt_uvd[:, 2] = (jt_uvd[:, 2] - center_xyz[2]) / (cube[2] / 2.0)
        jt_xyz = jt_xyz / (cube / 2.0)

        return (
            img[np.newaxis, :].astype(np.float32),
            jt_xyz.astype(np.float32),
            jt_uvd.astype(np.float32),
            center_xyz.astype(np.float32),
            M.astype(np.float32),
            cube.astype(np.float32),
        )

    def __len__(self):
        return len(self.data)

    def img_reader(self, img_path):
        img = Image.open(img_path)  # open image
        assert len(img.getbands()) == 1  # ensure depth image
        depth = np.asarray(img, np.float32)
        return depth

    def make_dataset(self):
        assert self.phase in ["train", "test"]
        center_refined_xyz, joints_xyz, joints_uvd, img_paths = self.read_joints(
            self.root
        )

        assert len(center_refined_xyz) == len(img_paths) == len(joints_xyz)

        item = list(zip(img_paths, joints_uvd, joints_xyz, center_refined_xyz))
        return item

    def read_joints(self, data_rt):
        centers_xyz, joints_xyz, joints_uvd, img_paths = [], [], [], []
        assert self.phase in ["train", "test"]

        center_path = "{}/center_{}_refined.txt".format(self.root, self.phase)
        anno_path = path.join(
            data_rt,
            self.phase,
            "test_annotation_frame.txt"
            if self.phase == "test"
            else "Training_Annotation.txt",
        )

        with open(anno_path) as f, open(center_path) as f_center:
            lines = [line.rstrip() for line in f]
            lines_center = [cline.rstrip() for cline in f_center]

            for index, line in enumerate(lines):
                strs = line.split()
                img_path = path.join(data_rt, self.phase, "images", strs[0][-19:])
                strs_center = lines_center[index].split()

                if not path.isfile(img_path) or strs_center[0] == "invalid":
                    continue

                joint_xyz = np.array(list(map(float, strs[1:]))).reshape(self.jt_num, 3)
                joint_uvd = xyz2uvd(joint_xyz, self.paras, self.flip)
                center_xyz = np.array(list(map(float, strs_center))).reshape(3)
                joints_xyz.append(joint_xyz)
                joints_uvd.append(joint_uvd)
                centers_xyz.append(center_xyz)
                img_paths.append(img_path)

        return centers_xyz, joints_xyz, joints_uvd, img_paths
