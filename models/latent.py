import torch
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    InterpolationMode,
    ToTensor,
    CenterCrop
)
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
from models.resnet18_32x32 import ResNet18_32x32

class LatentOOD(nn.Module):
    """OOD detection algorithm with pre-trained model
    MSP.

    fine-tuning (classification)
    classification
    outlier detection (ex. Mahalanobis) mean covariance computation
    MSP: maximum softmax probability
    MD: Mahalanobis distance
    """

    def __init__(
        self,
        backbone_name="vit_base_patch16_224",
        spherical=True,
        centercrop=False,
        n_class=None,
        pretrained=True,
        name=None,
    ):
        """
        spherical: project the representation to the unit sphere
        centercrop: use centercrop in preprocessing
        n_class: number of classes. None if no classification.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.supported_backbone = set([
            "resnet18.a1_in1k",# resnet18
            "vit_base_patch16_224",# vit_base_patch16_224.augreg2_in21k_ft_in1k
            "resnetv2_50x1_bitm",  # BiT
            "resnet50.a1_in1k", # resnet 50 trained with augmix method,
            "regnety_160.tv2_in1k", # regent for NN guide
            "vit_base_patch16_224.dino", # Linear classifier not pretrained (yet)
            "vit_base_patch16_clip_224.openai", # Linear classification only on 512 classes
            "resnet18_32x32"]) #resnet18_32x32
        assert backbone_name in self.supported_backbone, f"{backbone_name} not supported"
        self.spherical = spherical
        self.centercrop = centercrop
        self.n_class = n_class
        self.name = name

        # self.backbone = timm.create_model(
        #     backbone_name, pretrained=pretrained, num_classes=0
        # )  # num_classes=0 means feature extraction
        
        # self.classifier = timm.create_model(
        #     backbone_name, pretrained=pretrained, num_classes=n_class
        # ).head
        # print(f"new backbone")
        # print(f"head dim: {self.classifier}")

        if(self.backbone_name == "resnet18_32x32"):
            self.backbone = ResNet18_32x32(num_classes=self.n_class)
            self.classifier = self.backbone.get_fc_layer()
        
            if(pretrained):
                assert os.path.exists("./results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best_epoch99_acc0.7810.ckpt")
                state_dict = torch.load("./results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best_epoch99_acc0.7810.ckpt")
                self.backbone.load_state_dict(state_dict)
                self.classifier = self.backbone.get_fc_layer()

        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=n_class)  # num_classes=0 means feature extraction

            if backbone_name == 'resnet50.a1_in1k':
                self.backbone.load_state_dict(torch.load("datasets/imagenet_res50_v1.5/imagenet_resnet50_tvsv1_augmix_default/  ckpt.pth"))
    
            if n_class is not None:
                self.classifier = self.backbone.get_classifier()
                if self.classifier.weight.shape[0]!= n_class:
                    self.classifier = nn.Linear(self.backbone.num_features, n_class)
            else:
                self.classifier = None

        self.backbone.reset_classifier(0)

    def get_transform(self):
        config = resolve_data_config({}, model=self.backbone)
        transform = create_transform(**config)
        if self.centercrop:
            return transform
        else:
            if self.backbone_name == "vit_base_patch16_224":
                return Compose(
                    [
                        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                        ToTensor(),
                        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]
                )
            elif self.backbone_name in {"resnet18.a1_in1k", "resnet50.a1_in1k"}:
                return Compose(
                    [
                        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )
            elif self.backbone_name == "regnety_160.tv2_in1k":
                return Compose(
                    [
                        Resize((232, 232), interpolation=InterpolationMode.BILINEAR),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )
            else:
                raise NotImplementedError

    def forward(self, x):
        """anomaly score: maximum softmax probability"""
        logit = self.class_logit(x)
        prob = torch.softmax(logit, dim=1)
        return -prob.max(dim=1)[0]  # minus because we assign high score for anomaly

    def predict(self, x):
        """anomaly score"""
        return self(x)

    def class_logit(self, x):
        z = self.backbone(x)
        z = self._project(z)
        logit = self.classifier(z)
        return logit

    def encode(self, x):
        z = self.backbone(x)
        return self._project(z)

    def _project(self, x):
        if self.spherical:
            return x / x.norm(dim=1, keepdim=True)
        return x

    def train_step(self, x, y, opt, **kwargs):
        """
        fine-tuning with cross-entropy classification loss
        x: data
        y: label
        opt: optimizer
        """
        self.train()
        opt.zero_grad()
        logit = self.class_logit(x)
        loss = nn.CrossEntropyLoss()(logit, y)
        loss.backward()
        opt.step()
        d_train = {"loss": loss.item()}
        return d_train

    def validation_step(self, x, y, **kwargs):
        """
        return classification accuracy and loss
        """
        self.eval()
        logit = self.class_logit(x)
        loss = nn.CrossEntropyLoss()(logit, y)
        acc = (logit.argmax(dim=1) == y).float().mean().item()
        d_val = {"loss": loss.item(), "acc_": acc, "msp_":nn.functional.softmax(logit, dim=1).max().item()}
        return d_val
