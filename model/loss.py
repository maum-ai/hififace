import wandb

import torch

import torch.nn as nn
import torch.nn.functional as F
from .arcface_torch.backbones.iresnet import iresnet100
import lpips

from .Deep3DFaceRecon_pytorch.models.networks import ReconNetWrapper
from .Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel


class MultiScaleGANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(MultiScaleGANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input).detach()
        else:
            return torch.zeros_like(input).detach()

    def get_zero_tensor(self, input):
        return torch.zeros_like(input).detach()

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

class SIDLoss(nn.Module):
    def __init__(self):
        super(SIDLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, q_fuse, q_r, q_low, v_id_I_s, v_id_I_r, v_id_I_low):
        shape_loss = self.l1(q_fuse, q_r) + self.l1(q_fuse, q_low)
        inner_product_r = (torch.bmm(v_id_I_s.unsqueeze(1), v_id_I_r.unsqueeze(2)).squeeze())
        inner_product_low = (torch.bmm(v_id_I_s.unsqueeze(1), v_id_I_low.unsqueeze(2)).squeeze())
        id_loss = self.l1(torch.ones_like(inner_product_r), inner_product_r) + self.l1(torch.ones_like(inner_product_low), inner_product_low)

        sid_loss = 5 * id_loss + 0.5 * shape_loss

        return sid_loss, {"shape_loss": shape_loss,
                          "id_loss": id_loss,
                          "sid_loss": sid_loss,
                          }

class RealismLoss(nn.Module):
    def __init__(self):
        super(RealismLoss, self).__init__()
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        self.l1 = nn.L1Loss()

        self.adv_loss = MultiScaleGANLoss()

    def forward(self, m_tar, m_low, m_r, i_t, i_r, i_low, i_cycle, d_r, same):
        same = same.unsqueeze(-1).unsqueeze(-1)

        segmentation_loss = self.l1(F.interpolate(m_tar, scale_factor=0.25, mode='bilinear'), m_low) + self.l1(m_tar, m_r)
        reconstruction_loss = self.l1(i_r * same, i_t * same) + self.l1(i_low * same, F.interpolate(i_t, scale_factor=0.25, mode='bilinear') * same)
        cycle_loss = self.l1(i_t, i_cycle)
        lpips_loss = self.loss_fn_vgg(i_t * same, i_r * same).mean()
        adversarial_loss = self.adv_loss(d_r, True, for_discriminator=False)

        realism_loss = adversarial_loss + 100 * segmentation_loss + 20 * reconstruction_loss + cycle_loss + 5 * lpips_loss

        return realism_loss, {"segmentation_loss": segmentation_loss,
                              "reconstruction_loss": reconstruction_loss,
                              "cycle_loss": cycle_loss,
                              "lpips_loss": lpips_loss,
                              "adversarial_loss": adversarial_loss,
                              "realism_loss": realism_loss,
                              }

class GLoss(nn.Module):
    def __init__(self, f_3d_checkpoint_path, f_id_checkpoint_path, realism_config, sid_config):
        super(GLoss, self).__init__()
        self.f_3d = ReconNetWrapper(net_recon='resnet50', use_last_fc=False)
        self.f_3d.load_state_dict(torch.load(f_3d_checkpoint_path, map_location='cpu')['net_recon'])
        self.f_3d.eval()
        self.face_model = ParametricFaceModel()

        self.f_id = iresnet100(pretrained=False, fp16=False)
        self.f_id.load_state_dict(torch.load(f_id_checkpoint_path, map_location='cpu'))
        self.f_id.eval()

        self.realism_loss = RealismLoss(**realism_config)
        self.sid_loss = SIDLoss(**sid_config)

    def forward(self, i_s, i_t, i_r, i_low, i_cycle, m_tar, m_r, m_low, d_r, same):
        # region 3DMM
        with torch.no_grad():
            c_s = self.f_3d(F.interpolate(i_s, size=224, mode='bilinear'))
            c_t = self.f_3d(F.interpolate(i_t, size=224, mode='bilinear'))
        c_r = self.f_3d(F.interpolate(i_r, size=224, mode='bilinear'))
        c_low = self.f_3d(F.interpolate(i_low, size=224, mode='bilinear'))

        '''
        (B, 257)
        80 # id layer
        64 # exp layer
        80 # tex layer
        3  # angle layer
        27 # gamma layer
        2  # tx, ty
        1  # tz
        '''
        with torch.no_grad():
            c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)
            _, _, _, q_fuse = self.face_model.compute_for_render(c_fuse)

        _, _, _, q_r = self.face_model.compute_for_render(c_r)
        _, _, _, q_low = self.face_model.compute_for_render(c_low)
        # endregion

        # region arcface
        with torch.no_grad():
            v_id_i_s = F.normalize(self.f_id(F.interpolate((i_s - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2)

        v_id_i_r = F.normalize(self.f_id(F.interpolate((i_r - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2)
        v_id_i_low = F.normalize(self.f_id(F.interpolate((i_low - 0.5)/0.5, size=112, mode='bilinear')), dim=-1, p=2)
        # endregion

        sid_loss, sid_loss_dict = self.sid_loss(q_fuse, q_r, q_low, v_id_i_s, v_id_i_r, v_id_i_low)
        realism_loss, realism_loss_dict = self.realism_loss(m_tar, m_low, m_r, i_t, i_r, i_low, i_cycle, d_r, same)

        g_loss = sid_loss + realism_loss

        return g_loss, {**sid_loss_dict,
                        **realism_loss_dict,
                        "g_loss": g_loss,
                        }, \
               {"m_tar": m_tar,
                "m_r": m_r,
                "m_low": m_low,}



class DLoss(nn.Module):
    def __init__(self):
        super(DLoss, self).__init__()
        self.adv_loss = MultiScaleGANLoss()

    def forward(self, d_gt, d_fake):
        loss_real = self.adv_loss(d_gt, True)
        loss_fake = self.adv_loss(d_fake, False)

        d_loss = loss_real + loss_fake

        return d_loss, {"loss_real": loss_real,
                        "loss_fake": loss_fake,
                        "d_loss": d_loss,
                        }
