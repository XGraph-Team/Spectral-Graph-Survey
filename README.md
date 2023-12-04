# Awesome Spectral Graph Neural Networks
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 

## Contents

- [Survey Papers](#surveypapers)
- [Spatial and Spectral Views](#ssviews)
- [Application](#application)
  - [Computer Vision](#cv)
  - [Natural Language Processing](#nlp)
  - [Internet](#web)
  - [Recommender Systems](#rec)
  - [Healthcare](#health)
  - [Chemistry](#chemistry)
  - [Physics](#physics)
  - [Others](#others)


<a name="surveypapers" />

# Surveys
- [**Bridging the Gap between Spatial and Spectral Domains: A Unified Framework for Graph Neural Networks.**](https://dl.acm.org/doi/10.1145/3627816)
  - Zhiqian Chen, Fanglan Chen, Lei Zhang, Taoran Ji, Kaiqun Fu, Liang Zhao, Feng Chen, Lingfei Wu, Charu Aggarwal, Chang-Tien Lu.
  - **ACM Computer Surveys**, 2023
- [**A Survey on Spectral Graph Neural Networks.**](https://arxiv.org/abs/2302.05631)
  - Bo, Deyu, Xiao Wang, Yang Liu, Yuan Fang, Yawen Li, and Chuan Shi.
  - arXiv preprint arXiv:2302.05631 (2023)
- [**Transferability of spectral graph convolutional neural networks.**](https://jmlr.csail.mit.edu/papers/volume22/20-213/20-213.pdf)
  - Levie, Ron, Wei Huang, Lorenzo Bucci, Michael Bronstein, and Gitta Kutyniok.
  - **The Journal of Machine Learning Research** 22, no. 1 (2021): 12462-12520.
- [**Bridging the gap between spectral and spatial domains in graph neural networks.**](https://normandie-univ.hal.science/hal-02515637v1/file/DSGCN.pdf)
  - Balcilar, Muhammet, Guillaume Renton, Pierre Héroux, Benoit Gauzere, Sebastien Adam, Paul Honeine.
  - arXiv preprint arXiv:2003.11702 (2020). 


<a name="ssviews" />

# GNNs in Spatial and Spectral Views 

| Year        | Spatial Domain                                                                                                                                                                                                                                                                                                                                                                           | Spectral Domain                                                                                                                                                                                                                                                                                                        |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before 2015 | [ParWalk](https://proceedings.neurips.cc/paper_files/paper/2012/hash/512c5cad6c37edb98ae91c8a76c3a291-Abstract.html), [DeepWalk](https://dl.acm.org/doi/abs/10.1145/2623330.2623732), [LINE](https://arxiv.org/abs/1503.03578)                                                                                                                                                                                                             | [Spectral GNN](https://arxiv.org/abs/1312.6203), [ISGNN](https://arxiv.org/abs/1506.05163), [Neural graph fingerprints](https://proceedings.neurips.cc/paper_files/paper/2015/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html)                                                                              |
| 2016        | [DCNN](https://proceedings.neurips.cc/paper_files/paper/2016/hash/390e982518a50e280d8e2b535462ec1f-Abstract.html), [Molecular Graph Convolutions](https://link.springer.com/article/10.1007/s10822-016-9938-8), [PATCHY-SAN](http://proceedings.mlr.press/v48/niepert16)                   | [GCN](https://arxiv.org/abs/1609.02907), [ChebNet](https://proceedings.neurips.cc/paper_files/paper/2016/hash/04df4d434d481c5bb723be1b6df1ee65-Abstract.html)                                                                                                                                                                                                                                              |
| 2017        | [MPNN](https://arxiv.org/abs/1704.01212), [PGCN](https://proceedings.neurips.cc/paper/2017/hash/f507783927f2ec2737ba40afbd17efb5-Abstract.html), [GraphSAGE](https://proceedings.neurips.cc/paper_files/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)                                                                                                                                                                                                                                                                                          | [MoNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Monti_Geometric_Deep_Learning_CVPR_2017_paper.html)                                                                                                                                                                                                                                                                                       |
| 2018        | [GIN](https://arxiv.org/abs/1810.00826), [Adaptive GCN](https://proceedings.neurips.cc/paper_files/paper/2018/hash/01eee509ee2f68dc6014898c309e86bf-Abstract.html), [Fast GCN](https://arxiv.org/abs/1801.10247) , [JKNet](http://proceedings.mlr.press/v80/xu18c.html), [Large Scale GCN](https://dl.acm.org/doi/abs/10.1145/3219819.3219947)                                                                                                                                                                                                                        | [RationalNet](https://ieeexplore.ieee.org/abstract/document/8594830), [AR](https://ojs.aaai.org/index.php/AAAI/article/view/11604), [CayleyNet](https://ieeexplore.ieee.org/abstract/document/8521593)                                                                                                          |
| 2019        | [SGCN](https://proceedings.mlr.press/v97/wu19e.html), [DeepGCN](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_DeepGCNs_Can_GCNs_Go_As_Deep_As_CNNs_ICCV_2019_paper.html), [MixHop](http://proceedings.mlr.press/v97/abu-el-haija19a.html), [PPAP](https://arxiv.org/abs/1810.05997)                                                                                                                                                                                                                                                               | [ARMA](https://ieeexplore.ieee.org/abstract/document/9336270), [GDC](https://proceedings.neurips.cc/paper_files/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html), [EigenPool](https://dl.acm.org/doi/abs/10.1145/3292500.3330982), [GWNN](https://arxiv.org/abs/1904.07785), [Stable GCNN](https://dl.acm.org/doi/abs/10.1145/3292500.3330956)      |
| 2020        | [SIGN](https://arxiv.org/abs/2004.11198), [Spline GNN](https://ojs.aaai.org/index.php/AAAI/article/view/6185), [UaGGP](https://ojs.aaai.org/index.php/AAAI/article/view/5934), [GraLSP](https://ojs.aaai.org/index.php/AAAI/article/view/5861), [GraphSAINT](https://arxiv.org/abs/1907.04931), [DropEdge](https://arxiv.org/abs/1907.10903), [BGNN](https://arxiv.org/abs/2002.03575), [ALaGNN](https://dl.acm.org/doi/abs/10.5555/3491440.3491621), [Continuous GNN](http://proceedings.mlr.press/v119/xhonneux20a.html), [GCNII](https://proceedings.mlr.press/v119/chen20v.html), [PPRGo](https://dl.acm.org/doi/abs/10.1145/3394486.3403296), [DAGNN](https://dl.acm.org/doi/abs/10.1145/3394486.3403076), [H2GCN](https://proceedings.neurips.cc/paper/2020/hash/58ae23d878a47004366189884c2f8440-Abstract.html) | [GraphZoom](https://arxiv.org/abs/1910.02370)                                                                                                                                                                                                                                  |
| 2021        | [ADC](https://proceedings.neurips.cc/paper/2021/hash/c42af2fa7356818e0389593714f59b52-Abstract.html), [UGCN](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5857d68cd9280bc98d079fa912fd6740-Abstract.html), [DGC](https://proceedings.neurips.cc/paper/2021/hash/2d95666e2649fcfc6e3af75e09f5adb9-Abstract.html), [E(n)GNN](https://proceedings.mlr.press/v139/satorras21a.html), [GRAND](http://proceedings.mlr.press/v139/chamberlain21a.html), [C\&S](https://arxiv.org/abs/2010.13993), [LGNN](https://arxiv.org/abs/2110.14322)                                                                                                                                                                                          | [Interpretable Spectral Filter](http://proceedings.mlr.press/v139/kenlay21a.html), [Expressive Spectral Perspective](https://openreview.net/forum?id=-qh0M9XWxnv), [S2GC](https://openreview.net/forum?id=CYO5T-YjWZV), [BernNet](https://proceedings.neurips.cc/paper_files/paper/2021/hash/76f1cfd7754a6e4fc3281bcccb3d0902-Abstract.html)                                                                                                                                                      |
| 2022        | [GINR](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html), [Adaptive SGC](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ae07d152c51ea2ddae65aa7192eb5ff7-Abstract-Conference.html), [PGGNN](https://proceedings.mlr.press/v162/huang22l.html), [DIMP](https://ojs.aaai.org/index.php/AAAI/article/view/20353)                                                                                                                                                                                                                                                                        | [AGWN](https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.12), [ChebNetII](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f9b3ee2bcea04b327c09d7e3145bd1e-Abstract-Conference.html), [JacobiConv](https://proceedings.mlr.press/v162/wang22am.html), [SpecGNN](https://proceedings.mlr.press/v162/yang22n.html), [G2CN](https://proceedings.mlr.press/v162/li22h.html), [pGNN](https://proceedings.mlr.press/v162/fu22e.html), [ChebGibbsNet](https://openreview.net/forum?id=2a5Ru3JtNe0), [SpecFormer](https://arxiv.org/abs/2303.01028), [SIGN](https://arxiv.org/abs/2202.13013), [Spectral Density](https://dl.acm.org/doi/abs/10.1145/3488560.3498480)  |
| 2023        | [RSGNN](https://dl.acm.org/doi/abs/10.1145/3543507.3583221), [CAGCN](https://dl.acm.org/doi/abs/10.1145/3543507.3583229), [Low Rank GNN](https://dl.acm.org/doi/abs/10.1145/3543507.3583419), [Auto-HeG](https://arxiv.org/abs/2302.12357), [DropMessage](https://ojs.aaai.org/index.php/AAAI/article/view/25545)                                                                                                                                                                                                                                        | [DSF](https://dl.acm.org/doi/abs/10.1145/3543507.3583324), [F-SEGA](https://dl.acm.org/doi/abs/10.1145/3543507.3583423), [MidGCN](https://dl.acm.org/doi/abs/10.1145/3543507.3583335), [GHRN](https://dl.acm.org/doi/abs/10.1145/3543507.3583268)                                                                                                                                                                                                                  |


<a name="application" />

# Applications

# Experiment Code for Benchmarking Rational vs Polynomial


# Citation Info
```
@article{chen2023bridging,
  title        = {Bridging the Gap between Spatial and Spectral Domains: A Unified Framework for Graph Neural Networks},
  author       = {Chen, Zhiqian and Chen, Fanglan and Zhang, Lei and Ji, Taoran and Fu, Kaiqun and Zhao, Liang and Chen, Feng and Wu, Lingfei and Aggarwal, Charu and Lu, Chang-Tien},
  year         = 2023,
  journal      = {ACM Computing Surveys},
  doi          = {10.1145/3627816},
  url          = {https://dl.acm.org/doi/10.1145/3627816},
  publisher    = {Association for Computing Machinery (ACM)}
}
```
