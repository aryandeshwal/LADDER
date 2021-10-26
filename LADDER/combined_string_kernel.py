# gpytorch for substring kernel implemented in https://github.com/henrymoss/BOSS/tree/master/boss/code/kernels/string
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

import tensorflow as tf

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from weighted_retraining.opt_scripts.base import add_common_args, add_gp_args
from weighted_retraining.expr import expr_data
from weighted_retraining.train_scripts import train_expr
from weighted_retraining.utils import save_object, print_flush, DataWeighter
from weighted_retraining.gp_train import gp_train
from weighted_retraining.gp_opt import gp_opt

class CombinedStringKernel(Kernel):
        def __init__(self, base_latent_kernel, latent_train, train_smiles, model,  _gap_decay=1.0, _match_decay=1.0, _order_coefs=[1.0]*5, alphabet = [], maxlen=0, normalize=True, **kwargs):
            super().__init__(**kwargs)
            self.base_latent_kernel = base_latent_kernel # Kernel on the latent space (Matern Kernel)
            self.latent_train = latent_train # normalized training input
            self.lp_dim = self.latent_train.shape[-1]
            self.model = model # deep generative model (VAE)
            self.train_smiles = train_smiles # SMILES format training input #self.get_smiles(self.latent_train)#.clone())

            # Setting up parameters for string kernel
            self.register_parameter(
                name="raw_gap_decay",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
            )
            raw_gap_decay_constraint = Interval(0, 1)
            self.register_constraint("raw_gap_decay", raw_gap_decay_constraint)

            self.register_parameter(
                name="raw_match_decay",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
            )
            raw_match_decay_constraint = Interval(0, 1)
            self.register_constraint("raw_match_decay", raw_match_decay_constraint)

            self.register_parameter(
                name="raw_order_coefs",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, len(_order_coefs))),
            )
            raw_order_coefs_constraint = Interval(0, 1)
            self.register_constraint("raw_order_coefs", raw_order_coefs_constraint)

            self.alphabet = alphabet
            self.normalize = normalize
            try:
                self.embs, self.index = self.build_one_hot(self.alphabet)
            except Exception:
                raise Exception("check input alphabet covers X")
            self.embs_dim = self.embs.shape[1]
            self.maxlen = maxlen
        @property
        def gap_decay(self) -> torch.Tensor:
            return self.raw_gap_decay_constraint.transform(self.raw_gap_decay)

        @gap_decay.setter
        def gap_decay(self, value: torch.Tensor) -> None:
            if not torch.is_tensor(value):
                value = torch.tensor(value)

            self.initialize(raw_gap_decay=self.raw_gap_decay_constraint.inverse_transform(value))

        @property
        def match_decay(self) -> torch.Tensor:
            return self.raw_match_decay_constraint.transform(self.raw_match_decay)

        @match_decay.setter
        def match_decay(self, value: torch.Tensor) -> None:
            if not torch.is_tensor(value):
                value = torch.tensor(value)

            self.initialize(raw_match_decay=self.raw_match_decay_constraint.inverse_transform(value))

        @property
        def order_coefs(self) -> torch.Tensor:
            return self.raw_order_coefs_constraint.transform(self.raw_order_coefs)

        @order_coefs.setter
        def order_coefs(self, value: torch.Tensor) -> None:
            if not torch.is_tensor(value):
                value = torch.tensor(value)

            self.initialize(raw_order_coefs=self.raw_order_coefs_constraint.inverse_transform(value))
        
        def forward(self, z1, z2, **params):
            # z1 and z2 are unnormalized
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
            K_z1_training = self.string_forward(smiles_z1, self.train_smiles)
            K_z2_training = self.string_forward(smiles_z2, self.train_smiles)
            K_train_smiles = self.string_forward(self.train_smiles, self.train_smiles)
            K_train_smiles_inv = torch.inverse(K_train_smiles + 0.0001 * torch.eye(len(self.train_smiles)))
            kernel_val = K_z1_training @ K_train_smiles_inv.T @ (latent_space_kernel) @ K_train_smiles_inv  @ K_z2_training.T
            if check_dim > 0:
                kernel_val = kernel_val.unsqueeze(1)
            return kernel_val

        def string_forward(self, X, X2, **params):
            '''
                gpytorch implementation of string kernel from  https://github.com/henrymoss/BOSS
            '''
            check_dim = 0
            if len(X.shape) > 2:
                check_dim = X.squeeze(1)
                X = X.squeeze(1)
            if len(X2.shape) > 2:
                check_dim = X2.shape[0]
                X2 = X2[0]
            self.D = self._precalc()#.double()
            if self.normalize:
                X_diag_Ks = self._diag_calculations(X).double()
                X2_diag_Ks= self._diag_calculations(X2).double()
                k_results = torch.zeros((len(X), len(X2))).double()
            for i, x1 in enumerate(X):
                for j, x2 in enumerate(X2):
                    # if symmetric then only need to actually calc the upper gram matrix
                        k_result = self._k(x1,x2)#,self.embs, self.maxlen, self._gap_decay, self._match_decay, np.array(self._order_coefs),self.D,self.dD_dgap)
                        # Normalize kernel and gradients if required
                        if self.normalize and X_diag_Ks[i] != 0 and X2_diag_Ks[j] != 0:
                            k_result_norm = self._normalize(k_result, X_diag_Ks[i], X2_diag_Ks[j])
                            k_results[i, j] = k_result_norm
                            #print(f"k_result after norm {k_result_norm.dtype, k_results[i,j].dtype, X_diag_Ks[i], X2_diag_Ks[j].dtype}")
                        else:
                            k_results[i, j] = k_result
            #print(f"k_results {k_results}")
            if check_dim > 0:
                k_results = k_results.unsqueeze(1)
            return k_results

        def _k(self, s1, s2):#,  embs,  maxlen,  _gap_decay,  _match_decay, _order_coefs,D,dD_dgap):
            """
            TF code for vecotrized kernel calc.
            Following notation from Beck (2017), i.e have tensors S,D,Kpp,Kp

            Input is two np lists (of length maxlen) of integers represents each character in the alphabet
            calc kernel between these two string representations.
            """
            S = torch.from_numpy(self.embs[s1.int()] @ self.embs[s2.int()].T)
            Kp = torch.ones((len(self.order_coefs), self.maxlen, self.maxlen)).double()

            match_sq = self.match_decay * self.match_decay
            #print(f"S, Kp, match_sq {S.dtype, Kp.dtype, match_sq.dtype}")
            for i in range(len(self.order_coefs)-1):
                aux1 = S * Kp[i]
                #aux2 = np.dot(aux1,D)
                aux2 = aux1 @ self.D
                Kpp = match_sq * aux2
                Kp[i + 1] = (Kpp.T @ (self.D)).T 

            final_aux1 = S * Kp
            final_aux2 = torch.sum(final_aux1, dim=1)
            final_aux3 = torch.sum(final_aux2, dim=1)
            Ki = match_sq * final_aux3
            k = Ki.dot(self.order_coefs.double())
            #print(f" k {k.dtype}")
            return k #, dk_dgap, dk_dmatch, dk_dcoefs

        def _normalize(self, K_result, diag_Ks_i, diag_Ks_j):
            """
            Normalize the kernel and kernel derivatives.
            Following the derivation of Beck (2015)
            """
            norm = diag_Ks_i * diag_Ks_j
            sqrt_norm = torch.sqrt(norm)

            K_norm = K_result / sqrt_norm

            #print(f"norm, sqrt_norm, diag_Ks_i, K_norm {norm.dtype, sqrt_norm.dtype, diag_Ks_i.dtype, K_norm.dtype}")
            return K_norm#, gap_grads_norm, match_grads_norm, coef_grads_norm

        def _diag_calculations(self, X):
            """
            Calculate the K(x,x) values first because
            they are used in normalization.
            This is pre-normalization (otherwise diag is just ones)
            This function is not to be called directly, as requires preprocessing on X
            """
            # initialize return values
            k_result = torch.zeros(len(X))
            # All set up. Proceed with kernel matrix calculations
            for i, x1 in enumerate(X):
                result = self._k(x1,x1)#,self.embs, self.maxlen, self._gap_decay, self._match_decay, np.array(self._order_coefs),self.D,self.dD_dgap)
                k_result[i] = result#[0]
            return (k_result)#,gap_grads,match_grads,coef_grads)

        def _precalc(self):
            # Make D: a upper triangular matrix over decay powers.
            tril = torch.tril(torch.ones((self.maxlen,self.maxlen)))
            power = [[0]*i+list(range(0,self.maxlen-i)) for i in range(1,self.maxlen)]+[[0]*self.maxlen]
            power = (torch.tensor(power).reshape(self.maxlen,self.maxlen) + tril).double()

            tril = (tril.T - torch.eye(self.maxlen)).double()
            gaps = torch.ones([self.maxlen, self.maxlen])*self.gap_decay
            #print(f"gaps, tril, power {gaps.dtype, tril.dtype, power.dtype}")
            D = (gaps * tril) ** power
            #print(f"D {D.dtype}")
            return D#, dD_dgap



        # helper functions to perform operations on input strings

        def pad(self, s, length):
            #pad out input strings to our maxlen
            new_s = np.zeros(length)
            new_s[:len(s)] = s
            return new_s

        def encode_string(self, s, index):
            """
            Transform a string in a list of integers.
            The ints correspond to indices in an
            embeddings matrix.
            """
            return [index[symbol] for symbol in s]

        def build_one_hot(self, alphabet):
            """
            Build one-hot encodings for a given alphabet.
            """
            dim = len(alphabet)
            embs = np.zeros((dim+1, dim))
            index = {}
            for i, symbol in enumerate(alphabet):
                embs[i+1, i] = 1.0
                index[symbol] = i+1
            return embs, index

