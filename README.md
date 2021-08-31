# Code for the paper [Online Adversarial Attacks](https://arxiv.org/abs/2103.02014)

## Installation
To install package simply run:
`pip install -e .`

## Experiments
All the scripts to run the different experiments are in the `online_attacks/scripts` folder.

### Command to run toy experiment
`python -m online_attacks.experiments.toy --online_params.online_type stochastic_optimistic --online_params.N 100 --max_perms 1000 --K 10 `

### Command to run stochastic toy experiment
`python -m online_attacks.experiments.stochastic_toy --online_params.online_type stochastic_optimistic --online_params.N 100 --max_perms 1000 --K 10 --eps 5.0`

### Command to run the non-robust online attack experiment
`python -m online_attacks.scripts.random_eval --dataset mnist --attacker_type fgsm --name non_robust/mnist/fgsm --num_runs 10`

### Command to run the online attack on madry models:
You can compute the top-k indices selected by the online attacker:
`python -m online_attacks.scripts.online_attacks_sweep --dataset mnist --model_type madry --model_name adv_trained --attacker_type fgsm --name madry/mnist/fgsm --num_runs 10`

To compute the fool rate and competitive ratio you can run this command:
`python -m online_attacks.scripts.eval_all ./results/madry/mnist/fgsm --dataset mnist --model_type madry --model_name secret`

The results will be stored in json files under `./results/madry/mnist/fgsm/{uuid}/eval/madry/secret/`
The parameters used for the experiment are stored under `/results/madry/mnist/fgsm/{uuid}/eval/hparams.yaml`

### Visualizing the results
To visualize and explore the results you can use the `Dataframe` class:
```python
path = "./results/madry/mnist/fgsm/"
df = Dataframe()
df.aggregate_all(path)
stats = df.compute_statistics()
```

## Documentation
### Attacks
The different attacker are in the `online_attacks/attacks` folder.
Each attacker has a set of default parameters associated with it, those are the parameters that were used for the experiments.
To load an attacker call `create_attacker(classifier, attacker_type, attacker_params)`, you can add your own attack strategies by adding it to the attacks registry.

### Classifiers
The different classifiers used in the experiments are in the `online_attacks/classifiers` folder.
You can train your own classifier by calling:
`python -m online_attacks.scripts.train_classifiers --dataset mnist --model_type modelA`
To load a specific model you can use the following function: `load_classifier(dataset, model_type, model_name)`

### Online Algorithms
All the online algorithms are in the `online_attacks/online_algorithms`.
To instanciate the online algorithms use: `create_algorithm(algorithm_name, online_params, N)`

-------------
If you use this code please cite:
```
article{mladenovic2021online,
  title={Online Adversarial Attacks},
  author={Mladenovic, Andjela and Bose, Avishek Joey and Berard, Hugo and Hamilton, William L and Lacoste-Julien, Simon and Vincent, Pascal and Gidel, Gauthier},
  journal={arXiv preprint arXiv:2103.02014},
  year={2021}
}
```
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

### License
OnlineAttack is MIT licensed, as found in the LICENSE file.
