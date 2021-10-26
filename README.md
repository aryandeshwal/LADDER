This repository contains source code for the paper **Combining Latent Space and Structured Kernels for Bayesian Optimization over Combinatorial Spaces** 
accepted at NeurIPS-2021. 

- Running the code

  - Setting up conda environment from https://github.com/cambridge-mlg/weighted-retraining because we use their pre-existing VAE models. See 'Install dependencies' section.

  - Download datasets by following the 'Set up Data' section in the above repository. 

  - Relative imports are setup as if LADDER directory's code is in ``weighted-retraining/weighted_retraining/opt-scripts/``.

  - To run LADDER for chemical design task: ``python ladder_chemical_design.py``
	
  - To run LADDER for arithmetic expression optimization, some things need to be set up in above weighted-retraining environment
    - Add a comma in line 34 of weighted-retraining/weighted_retraining/train_scripts/train_expr.py
    - unzip eq2_grammar_dataset.zip in weighted-retraining/assets/data/expr
    - run ``python ladder_expression.py``

This code builds upon the existing [code](https://github.com/cambridge-mlg/weighted-retraining) provided by authors of 
*Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining*. We thank the authors for their code. 

Please consider citing both the papers if you use this work:

- Austin Tripp, Erik Daxberger, and Jos{\'e} Miguel Hern{\'a}ndez-Lobato. Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining, NeurIPS 2020.
- Aryan Deshwal and Janardhan Rao Doppa. Combining Latent Space and Structured Kernels for Bayesian Optimization over Combinatorial Spaces, NeurIPS 2021.






