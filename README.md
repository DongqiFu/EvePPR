# EvePPR: Everything Evolves in Personalized PageRank
This repository is for the WWW' 2023 paper "Everything Evolves in Personalized PageRank" ([Link](https://dongqifu.github.io/publications/EvePPR.pdf)). EvePPR and EvePPR-APP (approximated EvePPR) can efficiently and accurately track the Personalized PageRank vector after the transition matrix or stochastic vector has changed.



## Have a try
```bash
cd code
python main.py
```



## Full code for our experiment
In our research paper, we run three temporal graph alignment scenarios on movielens-1m, bitcoinalpha and wikilens datasets. The code can be downloaded from [Google Drive Link](https://drive.google.com/file/d/1S9BTumwhHqbM9UjYzoWZWX6qW3ICn1Jg/view?usp=drive_link) or [this repository](https://github.com/Violet24K/EvePPR-Full). The full code contains datasets and saved intermediate data (so that the user can simply np.load/sp.load instead of taking a long time recalculating everything). If you're not researching on temporal graph alignment topic, the simplified version in this repo should suffice.

The dataset we used in our experiments are processed [movielens-1m](https://github.com/Violet24K/Movielens-1M-Classified), [bitcoinalpha](https://github.com/Violet24K/Bitcoin-Alpha-Classified) and [wikilens](https://github.com/Violet24K/WikiLens-Classified).

Code for the baselines within our scenarios can be found in the repositories of this github account: [Violet24K](https://github.com/Violet24K).



## Reference
```
@inproceedings{DBLP:conf/www/LiFH23,
  author       = {Zihao Li and
                  Dongqi Fu and
                  Jingrui He},
  title        = {Everything Evolves in Personalized PageRank},
  booktitle    = {Proceedings of the {ACM} Web Conference 2023, {WWW} 2023, Austin,
                  TX, USA, 30 April 2023 - 4 May 2023},
  pages        = {3342--3352},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3543507.3583474},
  doi          = {10.1145/3543507.3583474},
  timestamp    = {Tue, 02 May 2023 14:07:23 +0200},
  biburl       = {https://dblp.org/rec/conf/www/LiFH23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
