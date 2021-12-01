import wandb

import pytorch_lightning as pl

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

from util import instantiate_from_config

class HifiFace(pl.LightningModule):
    def __init__(self, hp):
        super(HifiFace, self).__init__()
        self.hp = hp
        self.generator = instantiate_from_config(hp.generator)
        self.discriminator = instantiate_from_config(hp.discriminator)
        self.g_loss = instantiate_from_config(hp.g_loss)
        self.d_loss = instantiate_from_config(hp.d_loss)

        self.automatic_optimization = False

    @torch.no_grad()
    def interp(self, i_source1, i_source2, i_target, interp_rate=0.5, mode='all'):
        i_r, _, _, _ = self.generator.interp(i_source1, i_source2, i_target, interp_rate, mode)
        return i_r

    def forward(self, source_img, target_img):
        i_r, _, _, _ = self.generator(source_img, target_img)
        return i_r

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers(use_pl_optimizer=True)

        i_t = batch['target_image']
        i_s = batch['source_image']
        m_tar = batch['target_mask']
        same = batch['same']

        # region generator
        i_r, i_low, m_r, m_low = self.generator(i_s, i_t)
        i_cylce, _, _, _ = self.generator(i_t, i_r)

        d_r = self.discriminator(i_r)

        g_loss, g_loss_dict, image_dict = self.g_loss(i_s, i_t, i_r, i_low, i_cylce, m_tar, m_r, m_low, d_r, same)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        # endregion

        # region discriminator
        d_gt = self.discriminator(i_t)
        d_fake = self.discriminator(i_r.detach())

        d_loss, d_loss_dict = self.d_loss(d_gt, d_fake)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        # endregion

        # region logging
        self.logging_dict(g_loss_dict, prefix='train / ')
        self.logging_dict(d_loss_dict, prefix='train / ')
        self.logging_lr()

        #image logging
        if self.global_step % 1000 == 0:
            image_dict['I_target'] = i_t
            image_dict['I_source'] = i_s
            image_dict['I_low'] = i_low
            image_dict['I_r'] = i_r
            image_dict['I_cycle'] = i_cylce
            self.logging_image_dict(image_dict, prefix='train / ')
        # endregion

    def validation_step(self, batch, batch_idx):
        i_t = batch['target_image']
        i_s = batch['source_image']
        m_tar = batch['target_mask']
        same = batch['same']

        # region generator
        i_r, i_low, m_r, m_low = self.generator(i_s, i_t)
        i_cylce, _, _, _ = self.generator(i_t, i_r)

        d_r = self.discriminator(i_r)

        g_loss, g_loss_dict, image_dict = self.g_loss(i_s, i_t, i_r, i_low, i_cylce, m_tar, m_r, m_low, d_r, same)
        # endregion

        # region discriminator
        d_gt = self.discriminator(i_t)
        d_fake = self.discriminator(i_r.detach())

        d_loss, d_loss_dict = self.d_loss(d_gt, d_fake)
        # endregion

        # region logging
        self.logging_dict(g_loss_dict, prefix='validation / ')
        self.logging_dict(d_loss_dict, prefix='validation / ')

        image_dict['I_target'] = i_t
        image_dict['I_source'] = i_s
        image_dict['I_low'] = i_low
        image_dict['I_r'] = i_r
        image_dict['I_cycle'] = i_cylce
        # endregion

        return image_dict

    def validation_epoch_end(self, outputs):
        val_images = []

        for idx, output in enumerate(outputs):
            if idx > 30:
                break
            val_images.append(output['I_target'][0])
            val_images.append(output['I_source'][0])
            val_images.append(output['I_r'][0])
            val_images.append(output['I_cycle'][0])
            val_images.append(F.interpolate(output['I_low'], size=256, mode='bilinear')[0])
            val_images.append(output['m_tar'][0].repeat(3, 1, 1))
            val_images.append(output['m_r'][0].repeat(3, 1, 1))
            val_images.append(F.interpolate(output['m_low'].repeat(1, 3, 1, 1), size=256, mode='bilinear')[0])

        val_image = torchvision.utils.make_grid(val_images, nrow=8)
        self.logger.experiment.log({'validation / val_img': wandb.Image(val_image.clamp(0, 1))}, commit=False)

    def logging_dict(self, log_dict, prefix=None):
        for key, val in log_dict.items():
            if prefix is not None:
                key = prefix + key
            self.log(key, val)

    def logging_image_dict(self, image_dict, prefix=None, commit=False):
        for key, val in image_dict.items():
            if prefix is not None:
                key = prefix + key
            self.logger.experiment.log({key: wandb.Image(val.clamp(0, 1))}, commit=commit)

    def logging_lr(self):
        opts = self.trainer.optimizers
        for idx, opt in enumerate(opts):
            lr = None
            for param_group in opt.param_groups:
                lr = param_group['lr']
                break
            self.log(f"lr_{idx}", lr)


    def configure_optimizers(self):
        optimizer_list = []

        optimizer_g = instantiate_from_config(self.hp.generator.optimizer, params={"params": self.generator.parameters()})
        if "scheduler" in self.hp.generator:
            scheduler_g = instantiate_from_config(self.hp.generator.scheduler, params={"optimizer": optimizer_g})
            optimizer_list.append({
                "optimizer": optimizer_g,
                "lr_scheduler": {
                    "scheduler": scheduler_g,
                    "interval": 'step',
                    "monitor": False,
                },
            })
        else:
            optimizer_list.append({"optimizer": optimizer_g})

        optimizer_d = instantiate_from_config(self.hp.discriminator.optimizer, params={"params": self.discriminator.parameters()})
        if "scheduler" in self.hp.discriminator:
            scheduler_d = instantiate_from_config(self.hp.discriminator.scheduler, params={"optimizer": optimizer_g})
            optimizer_list.append({
                "optimizer": optimizer_d,
                "lr_scheduler": {
                    "scheduler": scheduler_d,
                    "interval": 'step',
                    "monitor": False,
                },
            })
        else:
            optimizer_list.append({"optimizer": optimizer_d})

        return optimizer_list

    def train_dataloader(self):
        trainset = instantiate_from_config(self.hp.dataset.train)
        return DataLoader(trainset, **self.hp.dataset.train.dataloader)

    def val_dataloader(self):
        valset = instantiate_from_config(self.hp.dataset.validation)
        return DataLoader(valset, **self.hp.dataset.validation.dataloader)