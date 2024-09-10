import torch
import torch.nn as nn
import torch.nn.functional as F

from xmuda.models.resnet34_unet import UNetResNet34
from xmuda.models.scn_unet import UNetSCN
from xmuda.models.CBAM import CBAMBlock
from xmuda.models.ExternalAttention import Mult_EA


class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs
                 ):
        super(Net2DSeg, self).__init__()

        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        self.proj_head1 = nn.Linear(80, feat_channels)
        self.proj_head2 = nn.Linear(feat_channels * 2, feat_channels)

        # segmentation head
        self.cls_head1 = nn.Linear(feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.cls_head2 = nn.Linear(feat_channels, num_classes)

    def forward(self, data_batch, feats_full=None, fusion_feats=None):
        if feats_full == None and fusion_feats == None:
            # (batch_size, 3, H, W)
            img = data_batch['img']

            # 2D network
            feats = self.net_2d(img)
            return feats
        else:
            # 2D-3D feature lifting
            img_indices = data_batch['img_indices']
            feats_2d = []
            for i in range(feats_full.shape[0]):
                feats_2d.append(feats_full.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
            feats_2d = torch.cat(feats_2d, 0)

            fusion_feats = F.relu(self.proj_head1(fusion_feats))  # 80-64
            fusion_feats = torch.cat([feats_2d, fusion_feats], dim=1)  # 128
            fusion_feats = self.proj_head2(fusion_feats)  # 128-64

            # linear
            x = self.cls_head1(fusion_feats)

            preds = {
                'seg_logit': x,
            }
            if self.dual_head:
                preds['seg_logit2'] = self.cls_head2(feats_2d)

            return preds


class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = UNetSCN(**backbone_3d_kwargs)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # segmentation head
        self.cls_head1 = nn.Linear(self.net_3d.out_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.cls_head2 = nn.Linear(self.net_3d.out_channels, num_classes)

    def forward(self, data_batch):
        feats = self.net_3d(data_batch['x'])

        x = self.cls_head1(feats)
        preds = {
            'seg_logit': x,
        }
        if self.dual_head:
            preds['seg_logit2'] = self.cls_head2(feats)

        return feats, preds


def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)


def test_Net3DSeg():
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)

    feats = feats.cuda()

    net_3d = Net3DSeg(num_seg_classes,
                      dual_head=True,
                      backbone_3d='SCN',
                      backbone_3d_kwargs={'in_channels': in_channels})

    net_3d.cuda()
    out_dict = net_3d({
        'x': [coords, feats],
    })
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)


class MFFM(nn.Module):
    def __init__(self):
        super(MFFM, self).__init__()
        self.cbam = CBAMBlock(channel=64, reduction=4, kernel_size=3)
        self.ea_2d = Mult_EA(dim=64)
        self.ea_3d = Mult_EA(dim=16)
        # self.ea_2d = Mult_EA(dim=64, num_heads=4)
        # self.ea_3d = Mult_EA(dim=16, num_heads=4)
        self.mlp = nn.Sequential(
            nn.Linear(80, 80),
            nn.ReLU(inplace=True),
        )
        self.ea_fuse = Mult_EA(dim=80)
        self.fusion = nn.Sequential(
            nn.Linear(80, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),
        )

    def forward(self, x2d, x3d, index):
        # image feature enhancement
        x2d = self.cbam(x2d)
        # 2D-3D feature lifting
        img_feats = []
        for i in range(x2d.shape[0]):
            img_feats.append(x2d.permute(0, 2, 3, 1)[i][index[i][:, 0], index[i][:, 1]])
        img_feats = torch.cat(img_feats, 0)
        img_feats = img_feats.unsqueeze(0)
        x_2d = (self.ea_2d(img_feats) + img_feats).squeeze(0)

        x_3d = x3d.unsqueeze(0)
        x_3d = (self.ea_3d(x_3d) + x_3d).squeeze(0)

        # 2D-3D feature fusion
        x_fusion = torch.cat([x_2d, x_3d], dim=1)
        x_fusion = self.mlp(x_fusion)
        x_fusion = x_fusion.unsqueeze(0)
        x_fusion = self.ea_fuse(x_fusion) + x_fusion
        x_fusion = x_fusion.squeeze(0)
        x_fusion = self.fusion(x_fusion)

        return x_fusion


class MFFM_v1(nn.Module):
    def __init__(self,num_classes):
        super(MFFM_v1, self).__init__()
        self.cbam = CBAMBlock(channel=64, reduction=4, kernel_size=3)
        self.ea_2d = Mult_EA(dim=64)
        self.ea_3d = Mult_EA(dim=16)
        self.mlp = nn.Sequential(
            nn.Linear(80, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, 80),
            nn.ReLU(inplace=True)
        )
        self.ea_fuse = Mult_EA(dim=80)
        self.fusion = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x2d, x3d, index):
        # image feature enhancement
        x2d = self.cbam(x2d)
        # 2D-3D feature lifting
        img_feats = []
        for i in range(x2d.shape[0]):
            img_feats.append(x2d.permute(0, 2, 3, 1)[i][index[i][:, 0], index[i][:, 1]])
        img_feats = torch.cat(img_feats, 0)
        img_feats = img_feats.unsqueeze(0)
        x_2d = (self.ea_2d(img_feats) + img_feats).squeeze(0)

        x_3d = x3d.unsqueeze(0)
        x_3d = (self.ea_3d(x_3d) + x_3d).squeeze(0)

        x_fusion = torch.cat([x_2d, x_3d], dim=1)
        x_fusion = self.mlp(x_fusion)
        x_fusion = x_fusion.unsqueeze(0)
        x_fusion = self.ea_fuse(x_fusion) + x_fusion
        x_fusion = x_fusion.squeeze(0)
        x_fusion = self.fusion(x_fusion)

        return x_fusion


def test_SELayer():
    x = torch.rand(1000, 1024)
    se = SELayer(1024)
    y = se(x)

def test_FusionNet():
    x2d = torch.rand(1000, 64)
    x3d = torch.rand(1000, 16)
    net = MFFM(5)
    y = net(x2d, x3d)


if __name__ == '__main__':
    # test_Net2DSeg()
    # test_Net3DSeg()
    # test_SELayer()
    test_FusionNet()
