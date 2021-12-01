import random

import torch
import torch.nn as nn

import torch.nn.functional as F

from .arcface_torch.backbones.iresnet import iresnet100

from .Deep3DFaceRecon_pytorch.models.networks import ReconNetWrapper
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample=False, up_sample=False):
        super(ResBlock, self).__init__()

        main_module_list = []
        main_module_list += [
            nn.InstanceNorm2d(in_channel),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            ]
        if down_sample:
            main_module_list.append(nn.AvgPool2d(kernel_size=2))
        elif up_sample:
            main_module_list.append(nn.Upsample(scale_factor=2, mode="bilinear"))

        main_module_list += [
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        ]
        self.main_path = nn.Sequential(*main_module_list)

        side_module_list = [nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)]
        if down_sample:
            side_module_list.append(nn.AvgPool2d(kernel_size=2))
        elif up_sample:
            side_module_list.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        self.side_path = nn.Sequential(*side_module_list)

    def forward(self, x):
        x1 = self.main_path(x)
        x2 = self.side_path(x)
        return x1 + x2

class AdaIn(nn.Module):
    def __init__(self, in_channel, vector_size):
        super(AdaIn, self).__init__()
        self.eps = 1e-5
        self.std_style_fc = nn.Linear(vector_size, in_channel)
        self.mean_style_fc = nn.Linear(vector_size, in_channel)

    def forward(self, x, style_vector):
        std_style = self.std_style_fc(style_vector)
        mean_style = self.mean_style_fc(style_vector)

        std_style = std_style.unsqueeze(-1).unsqueeze(-1)
        mean_style = mean_style.unsqueeze(-1).unsqueeze(-1)

        x = F.instance_norm(x)
        x = std_style * x + mean_style
        return x

class AdaInResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, up_sample=False):
        super(AdaInResBlock, self).__init__()
        self.vector_size = 257 + 512
        self.up_sample = up_sample

        self.adain1 = AdaIn(in_channel, self.vector_size)
        self.adain2 = AdaIn(out_channel, self.vector_size)

        main_module_list = []
        main_module_list += [
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        ]
        if up_sample:
            main_module_list.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        self.main_path1 = nn.Sequential(*main_module_list)

        self.main_path2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )

        side_module_list = [nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)]
        if up_sample:
            side_module_list.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        self.side_path = nn.Sequential(*side_module_list)


    def forward(self, x, id_vector):
        x1 = self.adain1(x, id_vector)
        x1 = self.main_path1(x1)
        x2 = self.side_path(x)

        x1 = self.adain2(x1, id_vector)
        x1 = self.main_path2(x1)

        return x1 + x2

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_first = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.channel_list = [64, 128, 256, 512, 512, 512, 512, 512]
        self.down_sample = [True, True, True, True, True, False, False]

        self.block_list = nn.ModuleList()

        for i in range(7):
            self.block_list.append(ResBlock(self.channel_list[i], self.channel_list[i+1], down_sample=self.down_sample[i]))

    def forward(self, x):
        x = self.conv_first(x)
        z_enc = None

        for i in range(7):
            x = self.block_list[i](x)
            if i == 1:
                z_enc = x
        return z_enc, x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.block_list = nn.ModuleList()
        self.channel_list = [512, 512, 512, 512, 512, 256]
        self.up_sample = [False, False, True, True, True]

        for i in range(5):
            self.block_list.append(AdaInResBlock(self.channel_list[i], self.channel_list[i+1], up_sample=self.up_sample[i]))


    def forward(self, x, id_vector):
        for i in range(5):
            x = self.block_list[i](x, id_vector)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, ):
        super(UpSamplingBlock, self).__init__()
        self.net = nn.Sequential(
            ResBlock(256, 64, up_sample=True),
            ResBlock(64, 16, up_sample=True),
            ResBlock(16, 8),
            ResBlock(8, 4),
        )

    def forward(self, x):
        x = self.net(x)
        m_r, i_r = x[:, 0, ...].unsqueeze(1), x[:, 1:, ...]
        m_r = F.tanh(m_r)
        return i_r, m_r

class SemanticFacialFusionModule(nn.Module):
    def __init__(self):
        super(SemanticFacialFusionModule, self).__init__()

        self.sigma = ResBlock(256, 256)
        self.low_mask_predict = ResBlock(256, 1)
        self.z_fuse_block = AdaInResBlock(256, 256 + 3)
        self.f_up = UpSamplingBlock()

    def forward(self, target_image, z_enc, z_dec, id_vector):
        z_enc = self.sigma(z_enc)
        m_low = self.low_mask_predict(z_dec)
        m_low = F.tanh(m_low)

        z_fuse = m_low * z_dec + (1 - m_low) * z_enc

        z_fuse = self.z_fuse_block(z_fuse, id_vector)

        i_low = z_fuse[:, 0:3, ...]
        z_fuse = z_fuse[:, 3:, ...]

        i_low = m_low * i_low + (1 - m_low) * F.interpolate(target_image, scale_factor=0.25)

        i_r, m_r = self.f_up(z_fuse)
        i_r = m_r * i_r + (1 - m_r) * target_image

        return i_r, i_low, m_r, m_low


class ShapeAwareIdentityExtractor(nn.Module):
    def __init__(self, f_3d_checkpoint_path, f_id_checkpoint_path):
        super(ShapeAwareIdentityExtractor, self).__init__()
        self.f_3d = ReconNetWrapper(net_recon='resnet50', use_last_fc=False)
        self.f_3d.load_state_dict(torch.load(f_3d_checkpoint_path, map_location='cpu')['net_recon'])
        self.f_3d.eval()
        self.f_id = iresnet100(pretrained=False, fp16=False)
        self.f_id.load_state_dict(torch.load(f_id_checkpoint_path, map_location='cpu'))
        self.f_id.eval()

    @torch.no_grad()
    def interp_all(self, i_source1, i_source2, i_target, interp_rate=0.5, mode='all'):
        mode_list = ['identity', '3d', 'all']
        assert interp_rate <= 1 and interp_rate >= 0, f"interpolation rate should be between 0 to 1, but got {interp_rate}"
        assert mode in mode_list, f"interpolation mode should be identity, 3d or all, but got {mode}"

        if mode == '3d' or mode == 'all':
            c_s1 = self.f_3d(i_source1)
            c_s2 = self.f_3d(i_source2)
            c_t = self.f_3d(i_target)
            c_interp = interp_rate * c_s1 + (1 - interp_rate) * c_s2
            c_fuse = torch.cat((c_interp[:, :80], c_t[:, 80:]), dim=1)
        else:
            c_s = self.f_3d(i_source1)
            c_t = self.f_3d(i_target)
            c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)

        if mode == 'identity' or mode == 'all':
            v_s = F.normalize(self.f_id(F.interpolate((i_source1 - 0.5) / 0.5, size=112, mode='bilinear')), dim=-1, p=2)
            v_t = F.normalize(self.f_id(F.interpolate((i_source2 - 0.5) / 0.5, size=112, mode='bilinear')), dim=-1, p=2)
            v_id = F.normalize(interp_rate * v_s + (1 - interp_rate) * v_t, dim=-1, p=2)
        else:
            v_id = F.normalize(self.f_id(F.interpolate((i_source1 - 0.5) / 0.5, size=112, mode='bilinear')), dim=-1, p=2)
            
        v_sid = torch.cat((c_fuse, v_id), dim=1)
        return v_sid

    @torch.no_grad()
    def forward(self, i_source, i_target):
        c_s = self.f_3d(i_source)
        c_t = self.f_3d(i_target)
        c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)

        v_id = F.normalize(self.f_id(F.interpolate((i_source - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2)

        v_sid = torch.cat((c_fuse, v_id), dim=1)
        return v_sid

class Generator(nn.Module):
    def __init__(self, identity_extractor_config):
        super(Generator, self).__init__()
        self.id_extractor = ShapeAwareIdentityExtractor(**identity_extractor_config)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sff_module = SemanticFacialFusionModule()

    @torch.no_grad()
    def interp(self, i_source1, i_source2, i_target, interp_rate=0.5, mode='all'):
        id_vector = self.id_extractor.interp_all(i_source1, i_source2, i_target, interp_rate, mode)
        z_enc, x = self.encoder(i_target)
        z_dec = self.decoder(x, id_vector)

        i_r, i_low, m_r, m_low = self.sff_module(i_target, z_enc, z_dec, id_vector)

        return i_r, i_low, m_r, m_low

    def forward(self, i_source, i_target):
        id_vector = self.id_extractor(i_source, i_target)
        z_enc, x = self.encoder(i_target)
        z_dec = self.decoder(x, id_vector)

        i_r, i_low, m_r, m_low = self.sff_module(i_target, z_enc, z_dec, id_vector)

        return i_r, i_low, m_r, m_low