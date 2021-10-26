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

from weighted_retraining import GP_TRAIN_FILE, GP_OPT_FILE
from weighted_retraining.chem.chem_data import (
    WeightedJTNNDataset,
    WeightedMolTreeFolder,
)
from weighted_retraining.chem.jtnn.datautils import tensorize
from weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining import utils
from weighted_retraining.chem.chem_utils import rdkit_quiet
from weighted_retraining.opt_scripts import base as wr_base

import gpytorch
import botorch
from botorch.models import FixedNoiseGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from gpytorch.kernels import ScaleKernel, CylindricalKernel, MaternKernel
from gpytorch.priors import GammaPrior
from botorch.optim import optimize_acqf
from gpytorch.kernels.kernel import Kernel

import time
import cma
#from combined_kernel import CombinedKernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.constraints import Interval, Positive
import torch
from data_utils import featurise_mols
from combined_fingerprints_kernel import *
import os



def _batch_decode_z_and_props(
    model: JTVAE,
    z: torch.Tensor,
    datamodule: WeightedJTNNDataset,
    args: argparse.Namespace,
    pbar: tqdm = None,
):
    """
    helper function to decode some latent vectors and calculate their properties
    """

    # Progress bar description
    if pbar is not None:
        old_desc = pbar.desc
        pbar.set_description("decoding")

    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            z_batch = z[j : j + batch_size]
            smiles_out = model.decode_deterministic(z_batch)
            if pbar is not None:
                pbar.update(z_batch.shape[0])
        z_decode += smiles_out

    # Now finding properties
    if pbar is not None:
        pbar.set_description("calc prop")

    # Calculate objective function values and choose which points to keep
    # Invalid points get a value of None
    z_prop = [
        args.invalid_score if s is None else datamodule.train_dataset.prop_func(s)
        for s in z_decode
    ]

    # Now back to normal
    if pbar is not None:
        pbar.set_description(old_desc)

    return z_decode, z_prop


def _encode_mol_trees(model, mol_trees):
    batch_size = 64
    mu_list = []
    with torch.no_grad():
        for i in range(0, len(mol_trees), batch_size):
            print(i)
            batch_slice = slice(i, i + batch_size)
            _, jtenc_holder, mpn_holder = tensorize(
                mol_trees[batch_slice], model.jtnn_vae.vocab, assm=False
            )
            tree_vecs, _, mol_vecs = model.jtnn_vae.encode(jtenc_holder, mpn_holder)
            muT = model.jtnn_vae.T_mean(tree_vecs)
            muG = model.jtnn_vae.G_mean(mol_vecs)
            mu = torch.cat([muT, muG], axis=-1).cpu().numpy()
            mu_list.append(mu)

    # Aggregate array
    mu = np.concatenate(mu_list, axis=0).astype(np.float32)
    return mu

rdkit_quiet()
#args = argparse.Namespace()
args = argparse.ArgumentParser()
args.seed = 0
args.query_budget = 500
args.retraining_frequency = 50
args.train_path = "data/chem/zinc/orig_model/tensors_train"
args.val_path="data/chem/zinc/orig_model/tensors_val"
args.vocab_file="data/chem/zinc/orig_model/vocab.txt"
args.property_file="data/chem/zinc/orig_model/pen_logP_all.pkl"
args.result_root="logs/opt/chem"
args.pretrained_model_file="assets/pretrained_models/chem.ckpt"
args.lso_strategy="opt"
args.n_retrain_epochs=0.1
args.n_init_retrain_epochs=1
args.n_best_points= 2000
args.n_rand_points= 8000
args.n_inducing_points=500
args.weight_type="rank"
args.rank_weight_k=1e-3


parser = WeightedJTNNDataset.add_model_specific_args(args)
parser = utils.DataWeighter.add_weight_args(parser)
parser = wr_base.add_common_args(parser)
parser = wr_base.add_gp_args(parser)
pl.seed_everything(parser.seed)
parser.weight_quantile =  0.8
parser.dbas_noise = 0.2
parser.rwr_alpha = 1e-1
parser.batch_size = 1
parser.property = "logP"
parser.train_path = "../../data/chem/zinc/orig_model/tensors_train"
parser.val_path="../../data/chem/zinc/orig_model/tensors_val"
parser.vocab_file="../../data/chem/zinc/orig_model/vocab.txt"
parser.property_file="../../data/chem/zinc/orig_model/pen_logP_all.pkl"
parser.result_root="../../logs/opt/chem"
parser.pretrained_model_file="../../assets/pretrained_models/chem.ckpt"
parser.invalid_score = -100


datamodule = WeightedJTNNDataset(parser, utils.DataWeighter(parser))
datamodule.setup("fit")

model = JTVAE.load_from_checkpoint(
    args.pretrained_model_file, vocab=datamodule.vocab
)
model.beta = model.hparams.beta_final  # Override any beta annealing
dset = datamodule.train_dataset


def initialize_model(train_x, train_obj, covar_module=None, state_dict=None):
    # define models for objective and constraint
    if covar_module is not None:
        model = SingleTaskGP(train_x, train_obj, covar_module=covar_module)
    else:
        model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def get_encoded_smiles(z, model):
    model = model.to(model.device)
    with torch.no_grad():
        decoded_smiles, _ = _batch_decode_z_and_props(
                model,
                torch.as_tensor(z, device=model.device),
                datamodule,
                parser)
    return decoded_smiles

def cma_es_concat(starting_point_for_cma, EI, model):
        es = cma.CMAEvolutionStrategy(x0=starting_point_for_cma, sigma0=0.2,inopts={'bounds': [-4,4], "popsize": 50},)
        iter = 1
        while not es.stop():
            iter += 1
            xs = es.ask()
            X = torch.tensor(xs)
            enc_smiles = torch.from_numpy(featurise_mols(get_encoded_smiles(X.float(), model), 'fingerprints')).double()
            concat_X = torch.cat([X, enc_smiles], axis=1).unsqueeze(1)
            Y = -1 * EI(concat_X)
            es.tell(xs, Y.detach().numpy())  # return the result to the optimizer
            print("current best")
            print(f"{es.best.f}")
            if (iter > 10):
                break
        return es.best.x, -1*es.best.f

for SEED in range(10):
    INIT_DATA_SIZE = 10
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"-------------- SEED: {SEED} --------------")
    # pick random_idxs to  the model
    random_idxs = np.random.choice(np.arange(len(dset)), size=INIT_DATA_SIZE, replace=False)
    #device = 'cuda:3'
    #model = model.to(device)
    mol_trees = [dset.data[i] for i in random_idxs]
    latent_points = _encode_mol_trees(model, mol_trees)
    targets = dset.data_properties[random_idxs]

    unnormalized_X_train = torch.from_numpy(latent_points).clone()
    with torch.no_grad():
        chosen_smiles, prop_val = _batch_decode_z_and_props(
                model,
                torch.as_tensor(unnormalized_X_train, device=model.device),
                datamodule,
                parser)
    for i in range(len(latent_points[0])):
        latent_points[:, i] = (latent_points[:, i] - np.min(latent_points[:, i]))/(np.max(latent_points[:, i]) - np.min(latent_points[:, i]))
    X_train = torch.from_numpy(latent_points)
    y_train = torch.from_numpy(targets).unsqueeze(-1)
    matern_kernel = MaternKernel(
                    nu=2.5,
                    ard_num_dims=X_train.shape[-1],
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                )
    all_encoded_smiles = torch.from_numpy(featurise_mols(chosen_smiles, 'fingerprints')).double()
    covar_module = ScaleKernel(base_kernel = CombinedFingerprintsKernel(base_latent_kernel=matern_kernel, latent_train=X_train.double(), train_smiles=all_encoded_smiles))
    concat_inputs = torch.cat([unnormalized_X_train, all_encoded_smiles], axis=1)
    gp_mll, gp_model = initialize_model(concat_inputs, y_train, covar_module=covar_module)
    dim = len(X_train[0])
    for i in range(len(X_train), 100):
        print(f"X_train shape {X_train.shape}")
        print(f"y_train shape {y_train.shape}")
        print(f"unnormalized_train shape {unnormalized_X_train.shape}")
        print(f"train_smiles shape {all_encoded_smiles.shape}")

        start_time = time.time()
        fit_gpytorch_model(gp_mll)
        print(f"Fitting done in {time.time()-start_time}")
        EI = ExpectedImprovement(gp_model, best_f = y_train.max().item())
        start_time = time.time()
        # Pick some random points first
        acq_points = np.random.randn(100, dim) 
        concat_acq_points = torch.cat([torch.from_numpy(acq_points), \
                                        torch.from_numpy(featurise_mols(get_encoded_smiles(torch.from_numpy(acq_points).float(), model), 'fingerprints')).double()], axis=1)
        ei_vals = EI(concat_acq_points.float().unsqueeze(1))
        starting_idxs = torch.argsort(-1*ei_vals)[:5]
        starting_points = np.concatenate([acq_points[starting_idxs], unnormalized_X_train[torch.argsort(-1*y_train.squeeze(1))[:5]].numpy()])
        best_points = []
        best_vals = []
        for starting_point_for_cma in starting_points:
            if (np.max(starting_point_for_cma) > 4 or np.min(starting_point_for_cma) < -4):
                continue
            newp, newv = cma_es_concat(starting_point_for_cma, EI, model)
            best_points.append(newp)
            best_vals.append(newv)
        print(f"best point {best_points[np.argmax(best_vals)]} \n with EI value {np.max(best_vals)}")
        print(f"Time for CMA-ES {time.time() - start_time}")
        for idx in np.argsort(-1*np.array(best_vals)):
            next_point = torch.from_numpy(best_points[idx]).float()
            if torch.all(next_point.unsqueeze(0) == unnormalized_X_train, axis=1).any():
                print(f"Point already in training data")
                continue
            with torch.no_grad():
                smiles, prop_val = _batch_decode_z_and_props(
                        model,
                        torch.as_tensor(next_point.unsqueeze(0), device=model.device),
                        datamodule,
                        parser)
            if smiles is None:
                print(f"Invalid decoding")
                continue
            chosen_smiles.append(smiles[0])
            all_encoded_smiles = torch.from_numpy(featurise_mols(chosen_smiles, 'fingerprints')).float()
            print(f"Found a good point")
            targets = np.concatenate([targets, prop_val])
            unnormalized_X_train = torch.cat([unnormalized_X_train, next_point.unsqueeze(0)])
            print(f"decoded prop_val:{prop_val}")
            print(f"decoded smile: {chosen_smiles[-1]}")
            break

        all_encoded_smiles = all_encoded_smiles.double()
        latent_points = unnormalized_X_train.clone()
        for j in range(len(latent_points[0])):
            latent_points[:, j] = (latent_points[:, j] - torch.min(latent_points[:, j]))/(torch.max(latent_points[:, j]) - torch.min(latent_points[:, j]))
        X_train = latent_points.double()
        y_train = torch.from_numpy((targets)).unsqueeze(-1)
        covar_module = ScaleKernel(base_kernel = CombinedFingerprintsKernel(base_latent_kernel=matern_kernel, latent_train=X_train, train_smiles=all_encoded_smiles))
        concat_inputs = torch.cat([unnormalized_X_train, all_encoded_smiles], axis=1)
        gp_mll, gp_model = initialize_model(concat_inputs, y_train, covar_module=covar_module)
        print(f"Best value found till now: {torch.max(y_train)}")
        torch.save({"smiles":chosen_smiles,   "latent_train_inputs_normalized":X_train, "unnormalized_X_train":unnormalized_X_train, "targets":targets, "y_train":y_train, "model_state_dict":gp_model.state_dict()}, "ladder_chemical_design_bo_nrun"+str(SEED)+".pkl")





