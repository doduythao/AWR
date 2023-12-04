import torch


class My_SmoothL1Loss(torch.nn.Module):
    def __init__(self):
        super(My_SmoothL1Loss, self).__init__()

    def forward(self, x, y):
        total_loss = 0
        assert x.shape == y.shape
        z = (x - y).float()
        mse_mask = (torch.abs(z) < 0.01).float()
        l1_mask = (torch.abs(z) >= 0.01).float()
        mse = mse_mask * z
        l1 = l1_mask * z
        total_loss += torch.mean(self._calculate_MSE(mse) * mse_mask)
        total_loss += torch.mean(self._calculate_L1(l1) * l1_mask)

        return total_loss

    def _calculate_MSE(self, z):
        return 0.5 * (torch.pow(z, 2))

    def _calculate_L1(self, z):
        return 0.01 * (torch.abs(z) - 0.005)


# to know to order of NYU 14 joints, read this
# https://github.com/mks0601/V2V-PoseNet_RELEASE/blob/master/src/data/NYU/data.lua
class AngleLoss(torch.nn.Module):
    def __init__(self, dataset):
        super(AngleLoss, self).__init__()
        hands17_pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 6),
            (6, 7),
            (7, 8),
            (2, 9),
            (9, 10),
            (10, 11),
            (3, 12),
            (12, 13),
            (13, 14),
            (4, 15),
            (15, 16),
            (16, 17),
            (5, 18),
            (18, 19),
            (19, 20),
        ]

        nyu_pairs = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (9, 10),
            (10, 13),
            (1, 13),
            (3, 13),
            (5, 13),
            (7, 13),
        ]

        self.pairs = hands17_pairs if dataset == "hands17" else nyu_pairs

    def forward(self, jt_uvd_pred, jt_uvd_gt):
        return torch.mean(
            torch.FloatTensor(
                [
                    self.calc_cosine_dist(jt_uvd_pred, jt_uvd_gt, pair)
                    for pair in self.pairs
                ]
            )
        )

    def calc_cosine_dist(self, points_pred, points_gt, pair):
        vector1 = points_pred[:, pair[0], :2] - points_pred[:, pair[1], :2]
        vector2 = points_gt[:, pair[0], :2] - points_gt[:, pair[1], :2]

        # Calculate dot product of direction vectors
        dot_product = torch.sum(vector1 * vector2, dim=1)

        magni1 = torch.norm(vector1, dim=1)
        magni2 = torch.norm(vector2, dim=1)

        # Calculate cosine distance
        return torch.mean(1 - torch.abs(dot_product / (magni1 * magni2)))


# don't use RatioLoss for NYU, only Hands17
class RatioLoss(torch.nn.Module):
    def __init__(self, dataset):
        super(RatioLoss, self).__init__()
        assert dataset == "hands17"
        self.jt_pairs = [
            (1, 6, 7),
            (6, 7, 8),
            (2, 9, 10),
            (9, 10, 11),
            (3, 12, 13),
            (12, 13, 14),
            (4, 15, 16),
            (15, 16, 17),
            (5, 18, 19),
            (18, 19, 20),
        ]

        self.limit = [
            (1.1232770882769216, 1.9434924208895799),
            (1.0157782929156474, 1.5776252413943375),
            (1.218069737439773, 2.224161733936492),
            (0.8576266902055922, 1.8000773993415229),
            (1.3092935932103715, 2.10393278074577),
            (1.0152851203388258, 1.9146196610416688),
            (1.2827101129324783, 2.2339825409413496),
            (0.9422953038839942, 1.7541158162409742),
            (1.2960763257299361, 2.2231989054240695),
            (0.7758295672892378, 1.6616790437060145),
        ]

    def forward(self, jt_uvd_pred, jt_uvd_gt):
        # only on predicted, no need gt.
        ratio_arr = [
            self.calc_ratio(
                jt_uvd_pred[:, pair[0], :2],
                jt_uvd_pred[:, pair[1], :2],
                jt_uvd_pred[:, pair[2], :2],
            )
            for pair in self.jt_pairs
        ]  # each element here is a ratio at spec pos (10 total)

        loss_arr = []
        for idx in range(10):
            x = ratio_arr[idx]
            min_val = self.limit[idx][0]
            max_val = self.limit[idx][1]
            l = torch.where(
                (x > min_val) & (x < max_val),
                torch.zeros_like(x),
                torch.where(x > max_val, x - max_val, min_val - x),
            )
            loss_arr.append(l.mean())
        return torch.mean(torch.stack(loss_arr))

    def calc_ratio(self, p1, p2, p3):
        # Calculate lengths
        p1_p2_length = torch.norm(p2 - p1, dim=1)
        p2_p3_length = torch.norm(p3 - p2, dim=1)

        # Calculate ratio
        return p1_p2_length / p2_p3_length
