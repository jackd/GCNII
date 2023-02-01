# Simpler Shallow Simple and Deep Graph Convolutional Networks (SS-GCNII)

This is a minimal extension of [GCNII](https://github.com/chennnM/GCNII) that explores the effectiveness of GCNII in the absence of variable transformations and non-linearities in propagation, and replacing the propagation mechanism entirely with a conjugate gradient personalized page rank solver. This is documented in [It's PageRank All The Way Down: Simplifying Deep Graph Networks (SDM23)](https://github.com/jackd/ppr-gnn-sdm23) (PPR-GNN).

## Dependencies

- pytorch-geometric
- networkx
- scikit-learn
- [torch_cg](https://github.com/sbarratt/torch_cg.git)

Example setup script using conda:

```bash
# Create/activate conda environment
conda create -n gcn2 python=3.7
conda activate gcn2

# Install dependencies from conda
conda install pyg -c pyg
conda install networkx scikit-learn

# Install torch_cg
git clone https://github.com/sbarratt/torch_cg.git
pip install torch_cg
```

## Semi Results

| Dataset  | GCNII        | SS-GCNII     | CG-GCNII     |
|----------|--------------|--------------|--------------|
| Cora     | 85.23 ± 0.57 | 85.15 ± 0.43 | 85.00 ± 0.39 |
| Citeseer | 73.14 ± 0.40 | 72.61 ± 1.17 | Hangs        |
| PubMed   | 80.32 ± 0.51 | 80.03 ± 0.33 | 80.01 ± 0.31 |

```bash
python -u train.py --data cora --layer 64 --test --seed 0
# acc = 0.8523 ± 0.005745432968889295
python -u train.py --data cora --layer 64 --test --simplified --seed 0
# acc = 0.8515 ± 0.004341658669218485
python -u train.py --data cora --layer 64 --test --dropout=0.7 --cg --seed 0
# acc = 0.85 ± 0.003872983346207421

python -u train.py --data citeseer --layer 32 --hidden 256 --lamda 0.6 --dropout 0.7 --test --seed 0
# acc = 0.7314 ± 0.003954743986657041
python -u train.py --data citeseer --layer 32 --hidden 256 --lamda 0.6 --dropout 0.7 --test --simplified --seed 0
# acc = 0.7261 ± 0.01167433081593971
python -u train.py --data citeseer --layer 32 --hidden 256 --lamda 0.6 --dropout 0.8 --test --cg --seed 0
# hangs?


python -u train.py --data pubmed --layer 16 --hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --test --seed 0
# acc = 0.8032 ± 0.005075431016179813
python -u train.py --data pubmed --layer 16 --hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --test --simplified --seed 0
# acc = 0.8003 ± 0.003287856444554722
python -u train.py --data pubmed --layer 16 --hidden 256 --lamda 0.4 --dropout 0.6 --wd1 5e-4 --test --cg --seed 0
# acc = 0.8002 ± 0.0031240998703626647
```

## Full Results

```bash
python -u full-supervised.py --data cora --layer 64 --alpha 0.2 --weight_decay 1e-4 --seed 0
# Test acc.:88.33 ± 1.18
python -u full-supervised.py --data cora --layer 64 --alpha 0.2 --weight_decay 1e-4 --seed 0 --simplified
# Test acc.:88.41 ± 1.15
python -u full-supervised.py --data cora --layer 64 --alpha 0.2 --weight_decay 1e-4 --seed 0 --cg --dropout 0.6
# Test acc.:88.61 ± 1.38

python -u full-supervised.py --data citeseer --layer 64 --weight_decay 5e-6 --seed 0
# Test acc.:77.13 ± 1.69
python -u full-supervised.py --data citeseer --layer 64 --weight_decay 5e-6 --seed 0 --simplified
# Test acc.:76.99 ± 1.65
python -u full-supervised.py --data citeseer --layer 64 --weight_decay 5e-6 --seed 0 --cg --dropout 0.6
# Test acc.:77.05 ± 1.84

python -u full-supervised.py --data pubmed --layer 64 --alpha 0.1 --weight_decay 5e-6 --seed 0
# Test acc.:89.57 ± 0.51
python -u full-supervised.py --data pubmed --layer 64 --alpha 0.1 --weight_decay 5e-6 --seed 0 --simplified
# Test acc.:87.38 ± 0.49
python -u full-supervised.py --data pubmed --layer 64 --alpha 0.1 --weight_decay 5e-6 --seed 0 --cg --dropout 0.6
# Test acc.:87.35 ± 0.51

python -u full-supervised.py --data chameleon --layer 8 --lamda 1.5 --alpha 0.2 --weight_decay 5e-4 --seed 0
# Test acc.:59.93 ± 2.74
python -u full-supervised.py --data chameleon --layer 8 --lamda 1.5 --alpha 0.2 --weight_decay 5e-4 --seed 0 --simplified
# Test acc.:53.84 ± 2.33
python -u full-supervised.py --data chameleon --layer 8 --lamda 1.5 --alpha 0.2 --weight_decay 5e-4 --seed 0 --cg --dropout 0.6
# Test acc.:42.52 ± 4.77

python -u full-supervised.py --data cornell --layer 16 --lamda 1 --weight_decay 1e-3 --seed 0
# Test acc.:75.41 ± 5.60
python -u full-supervised.py --data cornell --layer 16 --lamda 1 --weight_decay 1e-3 --seed 0 --simplified
# Test acc.:71.08 ± 6.05
python -u full-supervised.py --data cornell --layer 16 --lamda 1 --weight_decay 1e-3 --seed 0 --cg --dropout 0.6
# Test acc.:72.43 ± 8.09

python -u full-supervised.py --data texas --layer 32 --lamda 1.5 --weight_decay 1e-4 --seed 0
# Test acc.:69.73 ± 8.61
python -u full-supervised.py --data texas --layer 32 --lamda 1.5 --weight_decay 1e-4 --seed 0 --simplified
# Test acc.:65.68 ± 6.40
python -u full-supervised.py --data texas --layer 32 --lamda 1.5 --weight_decay 1e-4 --seed 0 --cg --dropout 0.6
# Test acc.:65.14 ± 5.60

python -u full-supervised.py --data wisconsin --layer 16 --lamda 1 --weight_decay 5e-4 --seed 0
# Test acc.:74.12 ± 5.60
python -u full-supervised.py --data wisconsin --layer 16 --lamda 1 --weight_decay 5e-4 --seed 0 --simplified
# Test acc.:70.59 ± 5.04
python -u full-supervised.py --data wisconsin --layer 16 --lamda 1 --weight_decay 5e-4 --seed 0 --cg --dropout 0.6
# Test acc.:68.63 ± 6.62
```

## Plot weights/pre-activations

To generate Figure 4 from [PPR-GNN](https://github.com/jackd/ppr-gnn-sdm23), use. Note some parameters like axis limits are hard coded and may not be suitable for all runs.

```bash
python -u train.py --data cora --layer 64 --test --seed 0 --repeats 1 --hist 0 1 63
```
