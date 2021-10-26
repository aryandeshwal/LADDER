from gpytorch.kernels.kernel import Kernel
from gpytorch.constraints import Interval, Positive
import torch
import numpy as np
import sys
import logging
import warnings
import itertools
import subprocess
from tqdm.auto import tqdm, trange
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import time
import pytorch_lightning as pl
from data_utils import transform_data, TaskDataLoader, featurise_mols



class CombinedFingerprintsKernel(Kernel):
        def __init__(self, base_latent_kernel, latent_train, train_smiles, **kwargs):
            super().__init__(**kwargs)
            self.base_latent_kernel = base_latent_kernel # Kernel on the latent space (Matern Kernel)
            self.latent_train = latent_train # normalized training input
            self.lp_dim = self.latent_train.shape[-1]
            self.train_smiles = train_smiles # SMILES format training input #self.get_smiles(self.latent_train)#.clone())

        def forward(self, z1, z2, **params):
            check_dim = 0
            if len(z1.shape) > 2:
                check_dim = z1.shape[0]
                z1 = z1.squeeze(1)
            if len(z2.shape) > 2:
                check_dim = z2.shape[0]
                z2 = z2[0]
            smiles_z1 = z1[:, self.lp_dim:]
            smiles_z2 = z2[:, self.lp_dim:]
            latent_space_kernel = self.base_latent_kernel.forward(self.latent_train, self.latent_train, **params)
            K_z1_training = self.fingerprint_forward(smiles_z1, self.train_smiles) 
            K_z2_training = self.fingerprint_forward(smiles_z2, self.train_smiles) 
            K_train_smiles = self.fingerprint_forward(self.train_smiles, self.train_smiles)
            K_train_smiles_inv = torch.inverse(K_train_smiles + 0.0001 * torch.eye(len(self.train_smiles))) # adding a nugget for stabilizing inverse
            kernel_val = K_z1_training @ K_train_smiles_inv.T @ (latent_space_kernel) @ K_train_smiles_inv  @ K_z2_training.T
            if check_dim > 0:
                kernel_val = kernel_val.unsqueeze(1)
            return kernel_val

        def fingerprint_forward(self, X, X2, **params):
            check_dim = 0
            EPS = 1e-6
            assert len(X.size()) == len(X2.size()), "inputs size not same"
            X_sqrd = torch.sum(torch.square(X), dim=-1, keepdim=True)
            X2_sqrd = torch.sum(torch.square(X2), dim=-1)
            cross_product = X @ X2.T#transpose(-1, -2)
            denominator = -cross_product + (X_sqrd + X2_sqrd)
            ret = (cross_product/denominator)
            return ret
