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

# My imports

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

from combined_string_kernel import *

import tensorflow as tf

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from weighted_retraining.opt_scripts.base import add_common_args, add_gp_args
from weighted_retraining.expr import expr_data
from weighted_retraining.train_scripts import train_expr
from weighted_retraining.utils import save_object, print_flush, DataWeighter
from weighted_retraining.gp_train import gp_train
from weighted_retraining.gp_opt import gp_opt

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


def pad(s, length):
    #pad out input strings to our maxlen
    new_s = np.zeros(length)
    new_s[:len(s)] = s
    return new_s


def encode_string(s, index):
    """
    Transform a string in a list of integers.
    The ints correspond to indices in an
    embeddings matrix.
    """
    return [index[symbol] for symbol in s]

def build_one_hot(alphabet):
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

args = argparse.ArgumentParser()
args.seed = 0
args.query_budget = 500
args.retraining_frequency = 50
args.root_dir="../../logs/opt/expr"
args.data_dir="../../assets/data/expr"
args.pretrained_model_file="../../assets/pretrained_models/expr/expr-k_inf.hdf5"
args.lso_strategy="opt"
args.n_retrain_epochs=1
args.n_init_retrain_epochs=1
args.n_data=500
args.n_best_points=2000
args.n_rand_points=8000
args.sn_inducing_points=500
args.samples_per_model=50
args.n_decode_attempts=5
args.ignore_percentile=50
args.weight_type="rank"
args.rank_weight_k=1e-3


parser = add_common_args(args)

parser = add_gp_args(parser)
parser = DataWeighter.add_weight_args(parser)
parser.latent_dim = 25
parser.result_root="../../logs/opt/expr"
parser.weight_quantile =  0.8
parser.dbas_noise = 0.2
parser.rwr_alpha = 1e-1
# parser.batch_size = 1
# parser.property = "logP"
# parser.train_path = "../../data/chem/zinc/orig_model/tensors_train"
# parser.val_path="../../data/chem/zinc/orig_model/tensors_val"
# parser.vocab_file="../../data/chem/zinc/orig_model/vocab.txt"
# parser.property_file="../../data/chem/zinc/orig_model/pen_logP_all.pkl"
# parser.result_root="../../logs/opt/chem"
# parser.pretrained_model_file="../../assets/pretrained_models/chem.ckpt"
parser.invalid_score = -100


tf.random.set_seed(parser.seed)
tf.config.experimental_run_functions_eagerly(True)
directory = Path(args.result_root)
directory.mkdir(parents=True, exist_ok=True)
opt_dir = directory / "opt{}".format(1)
opt_dir.mkdir(exist_ok=True)
data_weighter = DataWeighter(parser)

data_str, data_enc, data_scores = expr_data.get_initial_dataset_and_weights(
            Path(args.data_dir), args.ignore_percentile, args.n_data)

ite = 1
parser.retrain_from_scratch = False
model = train_expr.get_model(parser.pretrained_model_file, parser.latent_dim, parser.n_init_retrain_epochs,
                                     parser.n_retrain_epochs, parser.retrain_from_scratch, ite, opt_dir, data_enc, data_scores, data_weighter)



def get_encoded_smiles(z, index, maxlen):
    decoded_smiles = model.decode_from_latent_space(zs = tf.convert_to_tensor(z.numpy()), n_decode_attempts=50)
    decoded_smiles[np.argwhere(decoded_smiles == None)] = '+'
    all_convert_smiles = np.array([" ".join(list(smile)) for smile in decoded_smiles]).reshape(-1,1)
    all_convert_encoded_smiles = torch.tensor([[pad(encode_string(x[0].split(" "), index), maxlen)] for x in all_convert_smiles]).squeeze(1)
    return all_convert_encoded_smiles

def cma_es_concat(starting_point_for_cma, EI, index, maxlen):
        es = cma.CMAEvolutionStrategy(x0=starting_point_for_cma, sigma0=0.2,inopts={'bounds': [-4,4], "popsize": 50},)
        iter = 1
        while not es.stop():
            iter += 1
            xs = es.ask()
            X = torch.tensor(xs).float()#.unsqueeze(1)
            concat_X = torch.cat([X, get_encoded_smiles(X, index, maxlen)], axis=1).unsqueeze(1)
            with torch.no_grad():
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
    # Importing all string space related info
    random_idxs = np.random.choice(np.arange(10000), size=INIT_DATA_SIZE, replace=False)
    expr_data_stored = torch.load('expr_data_scores.pkl')
    data_str = expr_data_stored['data_str']
    data_scores = expr_data_stored['data_scores']
    chosen_smiles = [data_str[x] for x in random_idxs]
    targets = data_scores[random_idxs]
    # pick random_idxs to bootstrap the model
    latent_points = model.encode(chosen_smiles)
    unnormalized_X_train = torch.from_numpy(latent_points).clone()
    for i in range(len(latent_points[0])):
        latent_points[:, i] = (latent_points[:, i] - np.min(latent_points[:, i]))/(np.max(latent_points[:, i]) - np.min(latent_points[:, i]))

    all_expr_data = torch.load('expr_smiles.pkl')
    alphabet = all_expr_data['alphabet']
    embs = all_expr_data['embs']
    maxlen = all_expr_data['maxlen']
    index = all_expr_data['index']
    maxlen = 25
    all_smiles = np.array([" ".join(list(smile)) for smile in chosen_smiles]).reshape(-1,1)
    all_encoded_smiles = torch.tensor([[pad(encode_string(x[0].split(" "), index), maxlen)] for x in all_smiles]).squeeze(1).double()
    X_train = torch.from_numpy(latent_points).double()
    y_train = -1*torch.from_numpy(targets).unsqueeze(-1)
    matern_kernel = MaternKernel(
                    nu=2.5,
                    ard_num_dims=X_train.shape[-1],
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                )
    covar_module = ScaleKernel(base_kernel  = CombinedStringKernel(base_latent_kernel=matern_kernel, latent_train=X_train.double(), train_smiles=all_encoded_smiles,  \
                                                             model=model, alphabet=alphabet, maxlen=maxlen))
    concat_inputs = torch.cat([unnormalized_X_train, all_encoded_smiles], axis=1)
    gp_mll, gp_model = initialize_model(concat_inputs, y_train, covar_module=covar_module)

    dim = len(latent_points[0])
    for i in range(INIT_DATA_SIZE, 100):
        print(f"X_train shape {X_train.shape}")
        print(f"y_train shape {y_train.shape}")
        print(f"unnormalized_train shape {unnormalized_X_train.shape}")
        print(f"train_smiles shape {all_encoded_smiles.shape}")

        start_time = time.time()
        fit_gpytorch_model(gp_mll)#, options = {'maxiter':10})
        print(f"Fitting done in {time.time()-start_time}")
        start_time = time.time()
        EI = ExpectedImprovement(gp_model, best_f = y_train.max().item())
        # Pick some random points first
        acq_points = np.random.randn(100, dim) 
        concat_acq_points = torch.cat([torch.from_numpy(acq_points), get_encoded_smiles(torch.from_numpy(acq_points), index, maxlen )], axis=1)
        with torch.no_grad():
            ei_vals = EI(concat_acq_points.float().unsqueeze(1))
        starting_idxs = torch.argsort(-1*ei_vals)[:5]
        starting_points = np.concatenate([acq_points[starting_idxs], unnormalized_X_train[torch.argsort(-1*y_train.squeeze(1))[:5]].numpy()])
        best_points = []
        best_vals = []
        for starting_point_for_cma in starting_points:
            if (np.max(starting_point_for_cma) > 4 or np.min(starting_point_for_cma) < -4):
                continue
            newp, newv = cma_es_concat(starting_point_for_cma, EI,index, maxlen)
            best_points.append(newp)
            best_vals.append(newv)
        print(f"best point {best_points[np.argmax(best_vals)]} \n with EI value {np.max(best_vals)}")
        print(f"Time for CMA-ES {time.time() - start_time}")
        for idx in np.argsort(-1*np.array(best_vals)):
            next_point = tf.convert_to_tensor([best_points[idx]])
            smiles = model.decode_from_latent_space(zs=next_point, n_decode_attempts=20)
            if smiles[0] is None:
                print(f"Invalid decoding")
                continue
            chosen_smiles.append(smiles[0])
            try:
                all_smiles = np.array([" ".join(list(smile)) for smile in chosen_smiles]).reshape(-1,1)
                all_encoded_smiles = torch.tensor([[pad(encode_string(x[0].split(" "), index), maxlen)] for x in all_smiles]).squeeze(1)
            except:
                print(f"Invalid string output")
                del chosen_smiles[-1]
                continue

            print(f"Found a good point")
            prop_val = expr_data.score_function([chosen_smiles[-1]])
            targets = np.concatenate([targets, prop_val])
            unnormalized_X_train = torch.cat([unnormalized_X_train, torch.from_numpy(best_points[idx]).float().unsqueeze(0)])
            print(f"decoded prop_val:{prop_val}")
            print(f"decoded smile: {chosen_smiles[-1]}")
            break

        latent_points = unnormalized_X_train.clone()
        for j in range(len(latent_points[0])):
            latent_points[:, j] = (latent_points[:, j] - torch.min(latent_points[:, j]))/(torch.max(latent_points[:, j]) - torch.min(latent_points[:, j]))
        X_train = latent_points
        y_train = -1*torch.from_numpy((targets)).unsqueeze(-1)
        matern_kernel = MaternKernel(
                        nu=2.5,
                        ard_num_dims=X_train.shape[-1],
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                    )

        covar_module = ScaleKernel(base_kernel  = CombinedStringKernel(base_latent_kernel=matern_kernel, latent_train=X_train.double(), train_smiles=all_encoded_smiles,  \
                                                                 model=model, alphabet=alphabet, _order_coefs=[1.0]*5, maxlen=maxlen))
        concat_inputs = torch.cat([unnormalized_X_train, all_encoded_smiles], axis=1)
        gp_mll, gp_model = initialize_model(concat_inputs, y_train, covar_module=covar_module)#), state_dict=gp_model.state_dict())
        print(f"Best value found till now: {np.min(targets)}")
        torch.save({"smiles":chosen_smiles,   "latent_train_inputs_normalized":X_train, "unnormalized_X_train":unnormalized_X_train, "targets":targets, "y_train":y_train, "model_state_dict":gp_model.state_dict()}, "ladder_expression_bo_nrun"+str(SEED)+".pkl")
