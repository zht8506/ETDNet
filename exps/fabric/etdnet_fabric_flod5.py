# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp
import torch.nn as nn

dateset_path="/data/zht/fabric_dataset_5flod"



class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 20
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 20
        self.warmup_lr= 1e-6
        self.weight_decay=0.05

        self.dateset_path=dateset_path
        self.iouweigth = 5.0

        self.eval_interval=10


        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.eval_interval=10


        self.basic_lr_per_img= 1.6e-3 / 48
        self.min_lr_ratio=1e-5 / 1.6e-3
        self.max_epoch=111
        self.no_aug_epochs=30
        self.save_history_ckpt=False

        self.data_dir=dateset_path

        self.output_dir="./work_dir/fabric"
        self.train_ann = "train_5.json"
        self.val_ann = "val_5.json"

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                name="image",
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        # 分布式训练时的分布式加载数据
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        # 因为需要分布式，重写了sampler采样类
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        # 重新写了BatchSampler的方式，主要是将no_aug的是否去除马赛克增强的值传进去
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="image",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader


    # 需要使用adam优化器，将yolox原来的sgd换成adam优化器
    def get_optimizer(self, batch_size):
        from tools.set_optim import build_optimizer_lightvit
        dir_path=os.path.join(self.output_dir,"optim_parm.txt")
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            def get_parms(model):
                has_decay = []
                no_decay = []

                for name, param in model.named_parameters():

                    if not param.requires_grad:
                        continue  # frozen weights
                    if (len(param.shape) == 1 or name.endswith(".bias")):
                        no_decay.append(param)
                        # logger.info(f"{name} has no weight decay")
                    else:
                        has_decay.append(param)
                return [{'params': has_decay},
                        {'params': no_decay, 'weight_decay': 0.}]

            parameters=get_parms(self.model)

            optimizer = torch.optim.AdamW(parameters,
                                    betas=(0.9, 0.999),
                                    lr=lr,
                                    weight_decay=self.weight_decay)

            self.optimizer = optimizer

        return self.optimizer

    def get_model(self):
        from yolox.models import YOLOX
        from yolox.models.modified_LVT import M_LVT
        from yolox.models.cm_fpn import CM_FPN
        from yolox.models.tod_head import TOD_Head

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            out_channels= [64,160,256]
            backbone_noneck=M_LVT()
            backbone = CM_FPN(self.depth, self.width, out_channels=out_channels, act=self.act,backbone=backbone_noneck)
            head = TOD_Head(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)

        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model