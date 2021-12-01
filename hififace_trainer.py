import os
import sys
import wandb
import argparse
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from hififace_pl import HifiFace

sys.path.insert(0,'./model/Deep3DFaceRecon_pytorch')

parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str, required=True,
                    help="path of configuration yaml file about model")
parser.add_argument('--train_config', type=str, required=True,
                    help="path of configuration yaml file about training")
parser.add_argument('-g', '--gpus', type=str, default=None,
                    help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.")
parser.add_argument('-n', '--name', type=str, required=True,
                    help="Name of the run.")
parser.add_argument('-p', '--resume_checkpoint_path', type=str, default=None,
                    help="path of checkpoint for resuming")
parser.add_argument('--wandb_resume', type=str, default=None,
                    help="resume wandb logging from the input id")
args = parser.parse_args()

hp = OmegaConf.load(args.train_config)
save_path = os.path.join(hp.checkpoint.save_dir, args.name)
os.makedirs(save_path, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(hp.checkpoint.save_dir, args.name),
    **hp.checkpoint.callback
)

model_hparams = OmegaConf.load(args.model_config)
hififace_model = HifiFace(model_hparams)

if args.wandb_resume == None:
    resume = 'allow'
    wandb_id = wandb.util.generate_id()
else:
    resume = True
    wandb_id = args.wandb_resume

logger = WandbLogger(project=hp.wandb.project, entity=hp.wandb.entity, name=args.name,
                     config=OmegaConf.merge(hp, model_hparams), resume=resume, id=wandb_id)

logger.watch(hififace_model)

trainer = pl.Trainer(
    gpus=-1 if args.gpus is None else args.gpus,
    logger=logger,
    callbacks=[checkpoint_callback],
    weights_save_path=save_path,
    resume_from_checkpoint=args.resume_checkpoint_path,
    **hp.trainer
)

trainer.fit(hififace_model)
