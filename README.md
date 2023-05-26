# Official repository for 'Satellite-based high-resolution maps of cocoa for Côte d'Ivoire and Ghana'

This repository contains code for our Nature food paper [Cocoa plantations are associated with deforestation in Côte d’Ivoire and Ghana](https://www.nature.com/articles/s43016-023-00751-8).

<hr />

## Training
Simply run `python run.py launch` to start a single training run or `python run.py train` to start X training runs. You can set the number of different runs
in the provided `config.yaml` file. 
We also proivde a basic slurm launcher to start X jobs on a slurm cluster.

<hr />

## Data
Sadly, we cannot share our ground truth data due to copyright restrictions. We have included a dummy dataset showcasing what our dataloader expects as input.

<hr />

## Citation

Bibtex
```
@article{kalischek2023cocoa,
  title={Cocoa plantations are associated with deforestation in C{\^o}te d’Ivoire and Ghana},
  author={Kalischek, Nikolai and Lang, Nico and Renier, C{\'e}cile and Daudt, Rodrigo Caye and Addoah, Thomas and Thompson, William and Blaser-Hart, Wilma J and Garrett, Rachael and Schindler, Konrad and Wegner, Jan D},
  journal={Nature Food},
  volume={4},
  number={5},
  pages={384--393},
  year={2023},
  publisher={Nature Publishing Group}
}
```
or with DOI
```
Kalischek, N., Lang, N., Renier, C. et al. Cocoa plantations are associated with deforestation in Côte d’Ivoire and Ghana. 
Nat Food 4, 384–393 (2023). https://doi.org/10.1038/s43016-023-00751-8
```
