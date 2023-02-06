# PPR-GNN-TF

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains simplified/cleaned up tensorflow implementation of the work done for "It's PageRank All The Way Down: Simplifying Deep Graph Networks" to be presented as [SDM23](https://www.siam.org/conferences/cm/conference/sdm23).

We also include implementations of DAGNN and GCN2 (along with their static counterparts, SS-DAGNN and SS-GCN2 respectively).

See [this meta-repository](https://github.com/jackd/ppr-gnn-sdm23) for other sources.

## Setup

1. [Install tensorflow 2](https://www.tensorflow.org/install/).
2. Clone this repository and install as below.

```bash
git clone https://github.com/jackd/ppr-gnn-tf.git
cd ppr-gnn-tf
pip install -e .
```

## Results

This repo is a simplified/cleaned up reimplementation of the original work done in [graph-tf](https://github.com/jackd/graph-tf) which was used for numbers in the publication. Results from this repository are generally within 1 standard deviation of those reported, except in a few cases where the results in this repository are slightly better. The only intentional difference we are aware of is a change to the scale of the initial solution estimate `x0` of our conjugate gradient solver implementation.

Experiments with random splits in this repository use the same random splits as each other, but not necessarily the same splits as used in the original papers, though with the same number of train/validation examples.

Pytorch implementations for some models are available in other repositores:

- [jackd/DeeperGNN](https://github.com/jackd/DeeperGNN): DAGNN and SS-DAGNN implementations based on the [official DAGNN implementation](https://github.com/mengliu1998/DeeperGNN)
- [jackd/GCNII](https://github.com/jackd/GCNII): GCN2, SS-GCN2 and MLP-PPR (CG) implementations based on the [official GCN2 implementation](https://github.com/chennnM/GCNII)

See [scripts](#scripts) to generate results sourced from this repo.

### [Table 2: Small Citations Datasets](#table-2)

| Model          | Source          | Cora         | Citeseer     | PubMed       |
|----------------|-----------------|--------------|--------------|--------------|
| GCN2           | jackd/GCNII     | 85.23 ± 0.57 | 73.14 ± 0.40 | 80.32 ± 0.51 |
|                | This repo       | 84.84 ± 0.42 | 72.76 ± 1.10 | 80.22 ± 0.37 |
| SS-GCN2        | jackd/GCNII     | 85.15 ± 0.43 | 72.61 ± 1.17 | 80.03 ± 0.33 |
|                | This repo       | 85.06 ± 0.28 | 73.11 ± 0.97 | 79.90 ± 0.32 |
| DAGNN          | jackd/DeeperGNN | 84.15 ± 0.56 | 73.18 ± 0.50 | 80.62 ± 0.49 |
|                | This repo       | 84.13 ± 0.42 | 73.11 ± 0.57 | 80.12 ± 0.36 |
| SS-DAGNN       | jackd/DeeperGNN | 84.32 ± 0.64 | 73.08 ± 0.51 | 80.59 ± 0.47 |
|                | This repo       | 84.28 ± 0.47 | 73.12 ± 0.59 | 80.46 ± 0.65 |
| MLP-PPR (GCN2) | Paper           | 85.05 ± 0.29 | 72.86 ± 0.59 | 79.84 ± 0.25 |
|                | This repo       | 85.55 ± 0.43 | 72.81 ± 0.47 | 80.13 ± 0.32 |
| MLP-PPR (DAGNN)| Paper           | 84.71 ± 0.31 | 73.49 ± 0.75 | 80.47 ± 0.17 |
|                | This repo       | 84.53 ± 0.35 | 73.53 ± 0.55 | 80.62 ± 0.54 |
| PPR-MLP        | Paper           | 84.03 ± 0.53 | 72.87 ± 0.59 | 79.37 ± 0.27 |
|                | This repo       | 83.71 ± 0.42 | 72.38 ± 0.72 | 79.09 ± 0.49 |

### [Table 3: DAGNN Extras](#table-3)

| Model    | Source          | CS           | Physics      | Computer     | Photo        | ogbn-arxiv   |
|----------|-----------------|--------------|--------------|--------------|--------------|--------------|
| DAGNN    | jackd/DeeperGNN | 92.63 ± 0.53 | 94.03 ± 0.41 | 83.70 ± 1.45 | 91.32 ± 1.31 | 72.01 ± 0.26 |
|          | This repo       | 93.05 ± 0.36 | 94.34 ± 0.47 | 83.35 ± 1.48 | 92.14 ± 0.87 | 71.48 ± 0.25 |
| SS-DAGNN | jackd/DeeperGNN | 92.82 ± 0.34 | 93.83 ± 1.12 | 83.49 ± 1.12 | 91.26 ± 1.37 | 71.41 ± 0.24 |
|          | This repo       | 93.19 ± 0.41 | 94.34 ± 0.56 | 83.20 ± 1.94 | 92.17 ± 0.95 | 71.00 ± 0.25 |
| MLP-PPR  | Paper           | 93.29 ± 0.37 | 93.93 ± 0.59 | 82.88 ± 1.29 | 90.68 ± 1.40 | 71.50 ± 0.28 |
|          | This repo       | 93.29 ± 0.37 | 94.41 ± 0.52 | 82.65 ± 1.38 | 91.83 ± 0.51 | 71.42 ± 0.17 |
| PPR-MLP  | Paper           | 92.85 ± 0.41 | 93.82 ± 0.43 | 80.31 ± 1.24 | 90.70 ± 1.58 | 71.92 ± 0.23 |
|          | This repo       | 92.88 ± 0.49 | 94.38 ± 0.50 | 82.10 ± 1.60 | 90.93 ± 0.88 | 71.92 ± 0.34 |

### [Table 4: Bojchevski Comparison](#table-4)

| Model   | Source    | Cora-Full    | PubMed-random | Reddit       | MAG-Scholar-C |
|---------|-----------|--------------|---------------|--------------|---------------|
| PPR-MLP | Paper     | 62.97 ± 0.86 | 76.00 ± 2.27  | 26.28 ± 1.49 | 73.94 ± 1.90  |
|         | This repo | 63.66 ± 1.08 | 76.75 ± 2.56  | 27.80 ± 0.80 | 76.18 ± 0.89  |

### [Table 5: ogbn-papers100m](#table-5)

| Model      | Source    | Test Accuracy |
|------------|-----------|---------------|
| PPR-MLP    | Paper     | 66.21 ± 0.19  |
|            | This repo | 66.03 ± 0.16  |
| PPR-MLP XL | Paper     | 66.55 ± 0.28  |
|            | This repo | 66.45 ± 0.23  |

## Scripts

All experiments are configured using [gin-config](https://github.com/google/gin-config) and run using [gacl](https://github.com/jackd/gacl). For more straight-forward (though less flexible) scripts, see the [examples](./examples) directory.

With the exceptions experiments on `ogbn-papers100M` (data preprocessing) and `MAG-C-Scholar` / `mag-coarse` (data preprocessing and testing), all scripts below can be run with 4G GPU memory and 16G RAM with tensorflow 2.11.0 and python 3.7.15. Running these larger datasets requires ~256G and ~32G RAM respectively.

While efforts have been made to make these experiments deterministic, many use tensorflow's sparse dense matrix products during training/inference, hence the results are not deterministic when using GPUs. Parameter initialization and dropout sampling should be the same across runs however.

For tensorboard logging, simply add `--bindings="log_dir=/path/to/save"` to any of the below.

### Table 2

#### T2: GCN2

```bash
python -m ppr_gnn train/build-fit-test.gin gcn2/cora.gin
# Completed 10 trials
# test_acc           : 0.8484000325202942 ± 0.004199996732713463
# test_cross_entropy : 0.6859974086284637 ± 0.010212169457528942
# test_loss          : 1.1417751073837281 ± 0.01016093318982916
# val_acc            : 0.8201999008655548 ± 0.005758471372112476
# val_cross_entropy  : 0.7246695756912231 ± 0.00948237275705418
# val_loss           : 1.1804473042488097 ± 0.009525226582651983
python -m ppr_gnn train/build-fit-test.gin gcn2/citeseer.gin
# Completed 10 trials
# test_acc           : 0.7276000022888184 ± 0.01099270685746031
# test_cross_entropy : 1.2115771651268006 ± 0.0242442117192968
# test_loss          : 1.8351791262626649 ± 0.039361747433878914
# val_acc            : 0.7185999989509583 ± 0.009881302638391182
# val_cross_entropy  : 1.232261347770691 ± 0.02464872686223457
# val_loss           : 1.855863332748413 ± 0.039126087551023966
python -m ppr_gnn train/build-fit-test.gin gcn2/pubmed.gin
# Completed 10 trials
# test_acc           : 0.8021998882293702 ± 0.0036551143664345846
# test_cross_entropy : 0.553766405582428 ± 0.007415360663743098
# test_loss          : 0.8183935523033142 ± 0.017493113646583226
# val_acc            : 0.8120000481605529 ± 0.007429650006029639
# val_cross_entropy  : 0.5001193970441818 ± 0.00932245705326856
# val_loss           : 0.7647464811801911 ± 0.014367350031529112
```

#### T2: SS-GCN2

```bash
python -m ppr_gnn train/build-fit-test.gin gcn2/cora.gin --bindings='static=True'
# Completed 10 trials
# test_acc           : 0.8506000101566314 ± 0.0027640595158053643
# test_cross_entropy : 0.6820899546146393 ± 0.012254756239655315
# test_loss          : 1.11585214138031 ± 0.007398021601643033
# val_acc            : 0.8209999144077301 ± 0.004837355210916182
# val_cross_entropy  : 0.7249271512031555 ± 0.011652063691489667
# val_loss           : 1.1586893796920776 ± 0.006868321745143094
python -m ppr_gnn train/build-fit-test.gin gcn2/citeseer.gin --bindings='static=True'
# Completed 10 trials
# test_acc           : 0.7311000049114227 ± 0.009699981421616216
# test_cross_entropy : 1.2812947630882263 ± 0.022363429212371007
# test_loss          : 1.6579775094985962 ± 0.01385348135054478
# val_acc            : 0.7130000233650208 ± 0.007602627322747678
# val_cross_entropy  : 1.296779716014862 ± 0.020599996730667234
# val_loss           : 1.673462462425232 ± 0.012430823404609229
python -m ppr_gnn train/build-fit-test.gin gcn2/pubmed.gin --bindings='static=True'
# Completed 10 trials
# test_acc           : 0.7989998817443847 ± 0.0031937475574423807
# test_cross_entropy : 0.5176193833351135 ± 0.004003555228664387
# test_loss          : 0.7303836107254028 ± 0.007390049908973386
# val_acc            : 0.8096000552177429 ± 0.005642696954117608
# val_cross_entropy  : 0.4979209989309311 ± 0.0030003971433324823
# val_loss           : 0.7106851637363434 ± 0.004678967232935015
```

#### T2: DAGNN

```bash
python -m ppr_gnn train/build-fit-test.gin dagnn/cora.gin
# Completed 10 trials
# test_acc           : 0.8412999987602234 ± 0.004220168991673752
# test_cross_entropy : 0.7568712413311005 ± 0.010239873905934142
# test_loss          : 1.472684097290039 ± 0.01937362496623313
# val_acc            : 0.8197999238967896 ± 0.007820499251968196
# val_cross_entropy  : 0.7867293536663056 ± 0.009844716797018083
# val_loss           : 1.5025423049926758 ± 0.02036733159342149
python -m ppr_gnn train/build-fit-test.gin dagnn/citeseer.gin
# Completed 10 trials
# test_acc           : 0.7310999929904938 ± 0.005734989901353465
# test_cross_entropy : 1.1325032114982605 ± 0.006707983889196511
# test_loss          : 1.8679456830024719 ± 0.00850881976066861
# val_acc            : 0.7358000099658966 ± 0.011043521422436371
# val_cross_entropy  : 1.1558550238609313 ± 0.004990411699260933
# val_loss           : 1.8912974834442138 ± 0.010106794099884978
python -m ppr_gnn train/build-fit-test.gin dagnn/pubmed.gin
# Completed 10 trials
# test_acc           : 0.8011998772621155 ± 0.0036276773024886376
# test_cross_entropy : 0.5278672218322754 ± 0.006597766086544363
# test_loss          : 0.7516878962516784 ± 0.007961386268584728
# val_acc            : 0.8160000085830689 ± 0.00843800373980875
# val_cross_entropy  : 0.49183602929115294 ± 0.005063221979649717
# val_loss           : 0.7156566441059112 ± 0.0056769249142708785
```

#### T2: SS-DAGNN

```bash
python -m ppr_gnn train/build-fit-test.gin dagnn/cora.gin --bindings='static=True'
# Completed 10 trials
# test_acc           : 0.8428000092506409 ± 0.004728666548265752
# test_cross_entropy : 0.7343772768974304 ± 0.003626409795723298
# test_loss          : 1.4198065638542174 ± 0.006587055270019344
# val_acc            : 0.8163998901844025 ± 0.009112626398054822
# val_cross_entropy  : 0.7662060678005218 ± 0.004642915412504165
# val_loss           : 1.4516354560852052 ± 0.007984336980507798
python -m ppr_gnn train/build-fit-test.gin dagnn/citeseer.gin --bindings='static=True'
# Completed 10 trials
# test_acc           : 0.731199997663498 ± 0.005895767374858572
# test_cross_entropy : 1.1137439727783203 ± 0.0071064571580152286
# test_loss          : 1.8126948356628418 ± 0.005909009587227357
# val_acc            : 0.7324000060558319 ± 0.00966641517866798
# val_cross_entropy  : 1.1402317881584167 ± 0.005641412128265766
# val_loss           : 1.8391826748847961 ± 0.007837268916663961
python -m ppr_gnn train/build-fit-test.gin dagnn/pubmed.gin --bindings='static=True'
# Completed 10 trials
# test_acc           : 0.8045998811721802 ± 0.0065299457807338306
# test_cross_entropy : 0.528325766324997 ± 0.00808340448529347
# test_loss          : 0.7328670978546142 ± 0.02220860683697746
# val_acc            : 0.8194000422954559 ± 0.005730622909376554
# val_cross_entropy  : 0.49310185611248014 ± 0.006088665005724708
# val_loss           : 0.6976431429386138 ± 0.019824234341113873
```

#### T2: MLP-PPR (GCN2)

```bash
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/gcn2/cora.gin
# Completed 10 trials
# test_acc           : 0.8555000007152558 ± 0.004318558663606016
# test_cross_entropy : 0.7521181225776672 ± 0.007280258769613776
# test_loss          : 1.1594781637191773 ± 0.004582448991268499
# val_acc            : 0.8276000082492828 ± 0.007310265260590811
# val_cross_entropy  : 0.7815602779388428 ± 0.007416579864441596
# val_loss           : 1.1889203190803528 ± 0.004826897444484963
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/gcn2/citeseer.gin
# Completed 10 trials
# test_acc           : 0.7281000018119812 ± 0.004742363690282803
# test_cross_entropy : 1.019372570514679 ± 0.014657767770992546
# test_loss          : 1.6110474586486816 ± 0.010653960827004729
# val_acc            : 0.7263999998569488 ± 0.005425869270014344
# val_cross_entropy  : 1.0300427317619323 ± 0.012399432814541431
# val_loss           : 1.6217176079750062 ± 0.01131714336792094
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/gcn2/pubmed.gin
# Completed 10 trials
# test_acc           : 0.8013000071048737 ± 0.003195300381038667
# test_cross_entropy : 0.5146398067474365 ± 0.004967747413215709
# test_loss          : 0.7399441838264466 ± 0.005595875938050677
# val_acc            : 0.809200006723404 ± 0.004833209439189426
# val_cross_entropy  : 0.4923307776451111 ± 0.003493596779409087
# val_loss           : 0.7176351606845855 ± 0.003677843439182435
```

#### T2: MLP-PPR (DAGNN)

```bash
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/cora.gin
# Completed 10 trials
# test_acc           : 0.8453000009059906 ± 0.0034942943920666086
# test_cross_entropy : 0.598141485452652 ± 0.005916123470043148
# test_loss          : 1.1784679770469666 ± 0.006446560637862007
# val_acc            : 0.8204000055789947 ± 0.004799988865884738
# val_cross_entropy  : 0.6338410079479218 ± 0.00682026620075897
# val_loss           : 1.2141675233840943 ± 0.006645156897224646
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/citeseer.gin
# Completed 10 trials
# test_acc           : 0.7352999985218048 ± 0.0055145283408133995
# test_cross_entropy : 1.1569941878318786 ± 0.004809535115129386
# test_loss          : 1.835167157649994 ± 0.006862686624922901
# val_acc            : 0.7307999968528748 ± 0.009474173990457415
# val_cross_entropy  : 1.1783514261245727 ± 0.005021034239559213
# val_loss           : 1.856524395942688 ± 0.005780830791012097
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin
# Completed 10 trials
# test_acc           : 0.8062000036239624 ± 0.005362838937167564
# test_cross_entropy : 0.5207623422145844 ± 0.007687490747454096
# test_loss          : 0.731580662727356 ± 0.02014620049942169
# val_acc            : 0.8209999978542328 ± 0.009348791751222855
# val_cross_entropy  : 0.4802490085363388 ± 0.003489364834402499
# val_loss           : 0.6910673022270203 ± 0.012695903831069765
```

#### T2: PPR-MLP

```bash
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/cora.gin
# Completed 10 trials
# test_acc           : 0.8370999932289124 ± 0.004229653895614611
# test_cross_entropy : 0.8188188433647156 ± 0.006255727600176821
# test_loss          : 1.3798754930496215 ± 0.004347147410260704
# val_acc            : 0.8171999990940094 ± 0.0059464391487808405
# val_cross_entropy  : 0.8438221633434295 ± 0.0036963536666607517
# val_loss           : 1.4048787593841552 ± 0.0049385120739673825
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/citeseer.gin
# Completed 10 trials
# test_acc           : 0.7237999975681305 ± 0.007180531909935317
# test_cross_entropy : 1.2225730180740357 ± 0.00448654532005443
# test_loss          : 1.888606560230255 ± 0.011145342087714253
# val_acc            : 0.7282000005245208 ± 0.007820486904970038
# val_cross_entropy  : 1.2421355605125428 ± 0.002847040560379684
# val_loss           : 1.9081690907478333 ± 0.011602851365234762
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/pubmed.gin
# Completed 10 trials
# test_acc           : 0.7909000098705292 ± 0.004867238425412489
# test_cross_entropy : 0.5568277060985565 ± 0.005902854772228346
# test_loss          : 0.8863888263702393 ± 0.006664297456739514
# val_acc            : 0.8042000114917756 ± 0.007871466224676209
# val_cross_entropy  : 0.5339228510856628 ± 0.007439357431213176
# val_loss           : 0.8634839832782746 ± 0.006178381638626041
```

### Table 3

#### T3: DAGNN

```bash
python -m ppr_gnn train/build-fit-test.gin dagnn/cs.gin
# Completed 10 trials
# test_acc           : 0.9305124640464782 ± 0.003609063899309359
# test_cross_entropy : 0.2254168227314949 ± 0.01259587559271171
# test_loss          : 0.22541681975126265 ± 0.01259587616879692
# val_acc            : 0.9224444568157196 ± 0.012108556745934525
# val_cross_entropy  : 0.2507426306605339 ± 0.03655608275289787
# val_loss           : 0.25074262768030164 ± 0.036556080592186256
python -m ppr_gnn train/build-fit-test.gin dagnn/physics.gin
# Completed 10 trials
# test_acc           : 0.9433986723423005 ± 0.004712842514853302
# test_cross_entropy : 0.17836874425411225 ± 0.02156289137723504
# test_loss          : 0.17836874127388 ± 0.021562889169427595
# val_acc            : 0.9286666691303254 ± 0.01231081171403971
# val_cross_entropy  : 0.1892096295952797 ± 0.03192960076426989
# val_loss           : 0.18920962661504745 ± 0.03192960459448782
python -m ppr_gnn train/build-fit-test.gin dagnn/computer.gin
# Completed 10 trials
# test_acc           : 0.8334989607334137 ± 0.014787864934319007
# test_cross_entropy : 0.5964625239372253 ± 0.06874327007799773
# test_loss          : 0.7176997125148773 ± 0.06792456519196648
# val_acc            : 0.898000031709671 ± 0.020880624885149756
# val_cross_entropy  : 0.35276866853237154 ± 0.059819548896429056
# val_loss           : 0.47400584518909455 ± 0.06092971579401512
python -m ppr_gnn train/build-fit-test.gin dagnn/photo.gin
# Completed 10 trials
# test_acc           : 0.921405416727066 ± 0.008729032561273779
# test_cross_entropy : 0.3488931775093079 ± 0.02009329749859943
# test_loss          : 0.677765142917633 ± 0.020150056534972614
# val_acc            : 0.9337500333786011 ± 0.015189794079436034
# val_cross_entropy  : 0.32427197992801665 ± 0.036073176909071164
# val_loss           : 0.6531439781188965 ± 0.03372937555713657
python -m ppr_gnn train/build-fit-test.gin dagnn/ogbn-arxiv.gin
# Completed 10 trials
# test_acc           : 0.7148386180400849 ± 0.002498622580371116
# test_cross_entropy : 0.92684965133667 ± 0.015248378686146575
# test_loss          : 0.9268496334552765 ± 0.01524839987034911
# val_acc            : 0.7264203727245331 ± 0.0014528188413176586
# val_cross_entropy  : 0.8949163377285003 ± 0.01453646129511791
# val_loss           : 0.8949162900447846 ± 0.014536452993657903
```

#### T3: SS-DAGNN

```bash
python -m ppr_gnn train/build-fit-test.gin dagnn/cs.gin --bindings="static=True"
# Completed 10 trials
# test_acc           : 0.9318887770175934 ± 0.004099860892914089
# test_cross_entropy : 0.21805450767278672 ± 0.015131981724301616
# test_loss          : 0.21805450469255447 ± 0.01513198233865275
# val_acc            : 0.9224444389343261 ± 0.008575014007840389
# val_cross_entropy  : 0.24631571918725967 ± 0.03391633962043747
# val_loss           : 0.24631571620702744 ± 0.03391633874009664
python -m ppr_gnn train/build-fit-test.gin dagnn/physics.gin --bindings="static=True"
# Completed 10 trials
# test_acc           : 0.943360698223114 ± 0.005554185854242183
# test_cross_entropy : 0.17476900070905685 ± 0.020766431979182504
# test_loss          : 0.17476899772882462 ± 0.020766428558178604
# val_acc            : 0.9300000071525574 ± 0.012382800819303196
# val_cross_entropy  : 0.18801487535238265 ± 0.031455381282028225
# val_loss           : 0.18801487237215042 ± 0.031455384642206
python -m ppr_gnn train/build-fit-test.gin dagnn/computer.gin --bindings="static=True"
# Completed 10 trials
# test_acc           : 0.8320239126682282 ± 0.019383001022744328
# test_cross_entropy : 0.594784963130951 ± 0.0746083602186141
# test_loss          : 0.7166636765003205 ± 0.07274315173810532
# val_acc            : 0.8983333885669709 ± 0.020723047992452916
# val_cross_entropy  : 0.3512050032615662 ± 0.0609299057418584
# val_loss           : 0.47308371365070345 ± 0.06739574181541788
python -m ppr_gnn train/build-fit-test.gin dagnn/photo.gin --bindings="static=True"
# Completed 10 trials
# test_acc           : 0.9216876327991486 ± 0.009509361598893056
# test_cross_entropy : 0.3510319352149963 ± 0.022244528262830156
# test_loss          : 0.6730829954147339 ± 0.021430524500958108
# val_acc            : 0.9345833897590637 ± 0.01603492351519287
# val_cross_entropy  : 0.3245512515306473 ± 0.035267936340026285
# val_loss           : 0.6466022968292237 ± 0.03396173730657681
python -m ppr_gnn train/build-fit-test.gin dagnn/ogbn-arxiv.gin --bindings="static=True"
# Completed 10 trials
# test_acc           : 0.7099644184112549 ± 0.0018662815901742751
# test_cross_entropy : 0.9661507606506348 ± 0.005570077423505403
# test_loss          : 0.9661507606506348 ± 0.005570089998966085
# val_acc            : 0.7201114475727082 ± 0.0006929941780356842
# val_cross_entropy  : 0.9348713099956513 ± 0.003298422778202122
# val_loss           : 0.9348712503910065 ± 0.0032984225218535747
```

#### T3: MLP-PPR

```bash
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/cs.gin
# Completed 10 trials
# test_acc           : 0.9329466044902801 ± 0.0036728761822769555
# test_cross_entropy : 0.2122915878891945 ± 0.01373797658054346
# test_loss          : 0.2122915878891945 ± 0.01373797658054346
# val_acc            : 0.9242222249507904 ± 0.008632417277334435
# val_cross_entropy  : 0.23833443820476533 ± 0.03496099974997791
# val_loss           : 0.23833443820476533 ± 0.03496099974997791
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/physics.gin
# Completed 10 trials
# test_acc           : 0.9441111981868744 ± 0.005159544764429845
# test_cross_entropy : 0.17061436474323272 ± 0.020589107493233786
# test_loss          : 0.17061436474323272 ± 0.020589107493233786
# val_acc            : 0.9306666731834412 ± 0.014047524940360462
# val_cross_entropy  : 0.1799231708049774 ± 0.028721673525213456
# val_loss           : 0.1799231708049774 ± 0.028721673525213456
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/computer.gin
# Completed 10 trials
# test_acc           : 0.8265119135379791 ± 0.013817849711936739
# test_cross_entropy : 0.6035255670547486 ± 0.05197795994907993
# test_loss          : 0.7365565001964569 ± 0.05096195305128392
# val_acc            : 0.8963333308696747 ± 0.019405042739429067
# val_cross_entropy  : 0.3584261119365692 ± 0.05767449883173696
# val_loss           : 0.49145703911781313 ± 0.059318224456060095
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/photo.gin
# Completed 10 trials
# test_acc           : 0.9182728886604309 ± 0.005112853104196933
# test_cross_entropy : 0.3835767298936844 ± 0.01211591009853696
# test_loss          : 0.7499375581741333 ± 0.013440142179969913
# val_acc            : 0.9316666662693024 ± 0.017098568077878178
# val_cross_entropy  : 0.3524185538291931 ± 0.030372926564101267
# val_loss           : 0.7187793850898743 ± 0.028659333205636878
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/ogbn-arxiv.gin
# Completed 10 trials
# test_acc           : 0.7142295777797699 ± 0.0017281464517750885
# test_cross_entropy : 0.9395272672176361 ± 0.00511734535461177
# test_loss          : 0.9395272672176361 ± 0.00511734535461177
# val_acc            : 0.7237323343753814 ± 0.0010090056499000234
# val_cross_entropy  : 0.9046547114849091 ± 0.00347426013052122
# val_loss           : 0.9046547114849091 ± 0.00347426013052122
```

#### T3: PPR-MLP

```bash
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/cs.gin
# Completed 10 trials
# test_acc           : 0.9287664234638214 ± 0.004935144452523078
# test_cross_entropy : 0.23263700157403946 ± 0.020183086468951242
# test_loss          : 0.23263700157403946 ± 0.020183086468951242
# val_acc            : 0.9188888847827912 ± 0.008734767542325647
# val_cross_entropy  : 0.25546060502529144 ± 0.03556243964358238
# val_loss           : 0.25546060502529144 ± 0.03556243964358238
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/physics.gin
# Completed 10 trials
# test_acc           : 0.9438396215438842 +- 0.004907160529663477
# test_cross_entropy : 0.17871630787849427 +- 0.022142398700205533
# test_loss          : 0.17871630787849427 +- 0.022142398700205533
# val_acc            : 0.928000009059906 +- 0.013597372402103562
# val_cross_entropy  : 0.19266228526830673 +- 0.036039131438953105
# val_loss           : 0.19266228526830673 +- 0.036039131438953105
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/computer.gin
# Completed 10 trials
# test_acc           : 0.8210076808929443 ± 0.015961670552722462
# test_cross_entropy : 0.5897766947746277 ± 0.05412599540184989
# test_loss          : 0.7572716057300568 ± 0.053647419021392063
# val_acc            : 0.8849999964237213 ± 0.0189296843050888
# val_cross_entropy  : 0.3647976011037827 ± 0.049083379797668115
# val_loss           : 0.532292515039444 ± 0.0481335658154538
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/photo.gin
# Completed 10 trials
# test_acc           : 0.9092704951763153 ± 0.008846177722371335
# test_cross_entropy : 0.4046963691711426 ± 0.018211488800709774
# test_loss          : 0.7939214289188385 ± 0.016890830289765398
# val_acc            : 0.9200000107288361 ± 0.01622326814215438
# val_cross_entropy  : 0.3771723598241806 ± 0.03029853488468747
# val_loss           : 0.7663974046707154 ± 0.028180104094300683
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/ogbn-arxiv.gin
# Completed 10 trials
# test_acc           : 0.7191654860973358 ± 0.003366612029628019
# test_cross_entropy : 0.8917341530323029 ± 0.013429777759959842
# test_loss          : 0.8917341530323029 ± 0.013429777759959842
# val_acc            : 0.7299204766750336 ± 0.000523225378068322
# val_cross_entropy  : 0.8616855502128601 ± 0.008778855486464378
# val_loss           : 0.8616855502128601 ± 0.008778855486464378
```

### Table 4

#### T4: PPR-MLP

```bash
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/bojchevski/cora-full.gin
# Completed 10 trials
# test_acc           : 0.6366176545619965 ± 0.010754618674699
# test_cross_entropy : 1.3928760409355163 ± 0.033272564272832796
# test_loss          : 1.782378649711609 ± 0.0326026081607597
# val_acc            : 0.6368000090122223 ± 0.004075668182095769
# val_cross_entropy  : 1.3943402886390686 ± 0.013809487439239624
# val_loss           : 1.783842968940735 ± 0.013054618708322634
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/bojchevski/pubmed.gin
# Completed 10 trials
# test_acc           : 0.7675342440605164 ± 0.025648392764377383
# test_cross_entropy : 0.6000121057033538 ± 0.060618017754264106
# test_loss          : 0.6508087396621705 ± 0.059495381814306976
# val_acc            : 0.7658333361148835 ± 0.034763088346183406
# val_cross_entropy  : 0.6120973706245423 ± 0.07217964731440822
# val_loss           : 0.6628940343856812 ± 0.07106472831328986
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/bojchevski/reddit.gin
# Completed 10 trials
# test_acc           : 0.2780388057231903 ± 0.008019442622475243
# test_cross_entropy : 2.832725191116333 ± 0.06034908013850959
# test_loss          : 2.9526921987533568 ± 0.04831086421821603
# val_acc            : 0.27378048896789553 ± 0.009010042295523156
# val_cross_entropy  : 2.8545681715011595 ± 0.08441822898642877
# val_loss           : 2.9745352506637572 ± 0.07516816290909019
# Takes 32-64G memory, takes ~10 hours
python -m ppr_gnn train/build-fit-test-v2.gin ppr-mlp/bojchevski/mag-coarse.gin
# Completed 10 trials
# test_acc           : 0.7618487179279327 ± 0.008870135820222108
# test_cross_entropy : 0.796386057138443 ± 0.04655948230293778
# test_loss          : 0.7971788644790649 ± 0.04655050651128306
# val_acc            : 0.7841477274894715 ± 0.013153004905997981
# val_cross_entropy  : 0.7381663024425507 ± 0.06001596542550837
# val_loss           : 0.8118888974189759 ± 0.06609191387724059
```

### Table 5

```bash
# Creating data does not use GPU, but takes ~40 hours and ~256G ram.
python -m ppr_gnn ppr-mlp/create_cache.gin ppr-mlp/papers100m/base.gin
# -
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/papers100m/small.gin
# Completed 10 trials
# test_acc           : 0.6603182852268219 ± 0.0016392991952188156
# test_cross_entropy : 1.1794063806533814 ± 0.011220251860017226
# test_loss          : 1.1794063806533814 ± 0.011220251860017226
# val_acc            : 0.6910557568073272 ± 0.002181325848049496
# val_cross_entropy  : 0.9982379794120788 ± 0.01035559929037619
# val_loss           : 0.9982379794120788 ± 0.01035559929037619
python -m ppr_gnn train/build-fit-test.gin ppr-mlp/papers100m/large.gin # requires > 4G GPU memory
# Completed 10 trials
# test_acc           : 0.6645111858844757 ± 0.002338672424568246
# test_cross_entropy : 1.166238784790039 ± 0.01364191920210659
# test_loss          : 1.166238784790039 ± 0.01364191920210659
# val_acc            : 0.6964802622795105 ± 0.0031105457442762103
# val_cross_entropy  : 0.969533509016037 ± 0.015342101978564105
# val_loss           : 0.969533509016037 ± 0.015342101978564105
```

### Figure 6

The below train models to give the test accuracies (y-axis). For timings, replace `train/build-fit-test.gin` with `train/timings.gin`, e.g. the first becomes

```bash
python -m ppr_gnn train/timings.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 1e-0
'
# Warming up train step...: 100%|█████████████████| 10/10 [00:02<00:00,  4.66it/s]
# Benchmarking train step...: 100%|█████████████| 100/100 [00:01<00:00, 57.00it/s]
# Completed 100 train steps in 1754777μs, 17548μs / step
# Warming up validation_data test step...: 100%|██| 10/10 [00:00<00:00, 25.25it/s]
# Benchmarking validation_data test step...: 100%|█| 100/100 [00:01<00:00, 68.47it
# Completed 100 validation_data test steps in 1460891μs, 14609μs / step
# Warming up validation_data predict step...: 100%|█| 10/10 [00:00<00:00, 33.11it/
# Benchmarking validation_data predict step...: 100%|█| 100/100 [00:01<00:00, 70.3
# Completed 100 validation_data predict steps in 1422089μs, 14221μs / step
# Warming up test_data test step...: 100%|████████| 10/10 [00:00<00:00, 24.42it/s]
# Benchmarking test_data test step...: 100%|████| 100/100 [00:01<00:00, 64.39it/s]
# Completed 100 test_data test steps in 1553737μs, 15537μs / step
# Warming up test_data predict step...: 100%|█████| 10/10 [00:00<00:00, 30.57it/s]
# Benchmarking test_data predict step...: 100%|█| 100/100 [00:01<00:00, 67.58it/s]
# Completed 100 test_data predict steps in 1479986μs, 14800μs / step
```

#### CG

```bash
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 1e-0
'
# Completed 10 trials
# test_acc           : 0.7592000007629395 ± 0.012048233641971521
# test_cross_entropy : 0.612368780374527 ± 0.014936188231495413
# test_loss          : 0.8568277478218078 ± 0.05314952642223468
# val_acc            : 0.7867999970912933 ± 0.008494717518317572
# val_cross_entropy  : 0.5940962791442871 ± 0.00915246885402701
# val_loss           : 0.8385552525520324 ± 0.05249841851484844
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 5e-1
'
# Completed 10 trials
# test_acc           : 0.7868000090122222 ± 0.010916036585525455
# test_cross_entropy : 0.5545818686485291 ± 0.015323062779043023
# test_loss          : 0.7776873707771301 ± 0.02225242062858701
# val_acc            : 0.8060000002384186 ± 0.006511528619259823
# val_cross_entropy  : 0.5223732590675354 ± 0.009498405467158015
# val_loss           : 0.745478767156601 ± 0.014805424296696037
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 2e-1
'
# Completed 10 trials
# test_acc           : 0.804200005531311 ± 0.004354301978172176
# test_cross_entropy : 0.5283071339130402 ± 0.009018236775208487
# test_loss          : 0.7340270757675171 ± 0.01974434259957585
# val_acc            : 0.8178000032901764 ± 0.005895763533062925
# val_cross_entropy  : 0.49004908800125124 ± 0.004664872276535674
# val_loss           : 0.6957690358161926 ± 0.013833453664829309
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 1e-1
'
# Completed 10 trials
# test_acc           : 0.8094000041484832 ± 0.004363494781523124
# test_cross_entropy : 0.5201887130737305 ± 0.005044494784566693
# test_loss          : 0.733194887638092 ± 0.014064456569640674
# val_acc            : 0.8236000120639801 ± 0.008236505462867267
# val_cross_entropy  : 0.482197979092598 ± 0.004914012978008159
# val_loss           : 0.6952041685581207 ± 0.01358429844430118
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 5e-2
'
# Completed 10 trials
# test_acc           : 0.8044000089168548 ± 0.005678021277934181
# test_cross_entropy : 0.5180065333843231 ± 0.007078628732602231
# test_loss          : 0.7273494899272919 ± 0.016243860376347355
# val_acc            : 0.8229999959468841 ± 0.01077960928112771
# val_cross_entropy  : 0.48360424041748046 ± 0.0037559579514049304
# val_loss           : 0.6929471850395202 ± 0.011477244423552488
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 2e-2
'
# Completed 10 trials
# test_acc           : 0.8066999971866607 ± 0.004627095865234546
# test_cross_entropy : 0.5201169788837433 ± 0.0077785899692975665
# test_loss          : 0.7309369027614594 ± 0.015215114594295587
# val_acc            : 0.8194000005722046 ± 0.007158214833098513
# val_cross_entropy  : 0.4834901511669159 ± 0.0032132723614076097
# val_loss           : 0.6943100750446319 ± 0.01041815505223871
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 1e-2
'
# Completed 10 trials
# test_acc           : 0.8038000047206879 ± 0.004512204906115583
# test_cross_entropy : 0.5196862280368805 ± 0.006978845324554642
# test_loss          : 0.7305161833763123 ± 0.01608326291708329
# val_acc            : 0.8200000047683715 ± 0.007483315601932721
# val_cross_entropy  : 0.48327924907207487 ± 0.003095633990554271
# val_loss           : 0.6941092133522033 ± 0.010205903869982248
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 5e-3
'
# Completed 10 trials
# test_acc           : 0.8048999965190887 ± 0.004060778092760142
# test_cross_entropy : 0.5204649150371552 ± 0.007817540603396796
# test_loss          : 0.731310898065567 ± 0.017085697362950333
# val_acc            : 0.8176000058650971 ± 0.009499461046302974
# val_cross_entropy  : 0.4828063189983368 ± 0.0029561877979135736
# val_loss           : 0.6936523139476776 ± 0.01033659479753367
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 2e-3
'
# Completed 10 trials
# test_acc           : 0.8048999905586243 ± 0.0035623081303218667
# test_cross_entropy : 0.5182217240333558 ± 0.007029198604891675
# test_loss          : 0.7314444720745087 ± 0.019863040719763053
# val_acc            : 0.8198000073432923 ± 0.007180535064260815
# val_cross_entropy  : 0.48194176852703097 ± 0.003925784670970116
# val_loss           : 0.6951645374298095 ± 0.014060253018701833
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin --bindings='
preprocess_train = False
tol = 1e-3
'
# Completed 10 trials
# test_acc           : 0.8049000084400177 ± 0.004300001471579183
# test_cross_entropy : 0.5187098145484924 ± 0.007309202163799101
# test_loss          : 0.7278119802474976 ± 0.015237307205686118
# val_acc            : 0.8189999997615814 ± 0.009176047436354086
# val_cross_entropy  : 0.4823420763015747 ± 0.00365429216958969
# val_loss           : 0.6914442598819732 ± 0.01020443552417974
```

#### Low Rank

```bash
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin propagators/ppr/low-rank.gin --bindings='
preprocess_train = False
rank = 32
'
# Completed 10 trials
# test_acc           : 0.7132000029087067 ± 0.00625778581823172
# test_cross_entropy : 0.703421700000763 ± 0.0012168840405256537
# test_loss          : 0.7852751851081848 ± 0.0045369385279902625
# val_acc            : 0.7148000001907349 ± 0.005075430027850927
# val_cross_entropy  : 0.6888110935688019 ± 0.0009435872922054709
# val_loss           : 0.7706645905971528 ± 0.004028357133796872
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin propagators/ppr/low-rank.gin --bindings='
preprocess_train = False
rank = 64
'
# Completed 10 trials
# test_acc           : 0.7610000014305115 ± 0.005000005960481389
# test_cross_entropy : 0.662298995256424 ± 0.0075780815367138665
# test_loss          : 0.7797136962413788 ± 0.013751399366549396
# val_acc            : 0.7685999929904938 ± 0.006003321241846037
# val_cross_entropy  : 0.6399376571178437 ± 0.0034407502930335733
# val_loss           : 0.7573523581027984 ± 0.013436167712558426
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin propagators/ppr/low-rank.gin --bindings='
preprocess_train = False
rank = 128
'
# Completed 10 trials
# test_acc           : 0.7589000105857849 ± 0.006284110543179483
# test_cross_entropy : 0.6437891066074372 ± 0.006347319347680296
# test_loss          : 0.7818643450737 ± 0.007362994100632403
# val_acc            : 0.7795999944210052 ± 0.009583314378366799
# val_cross_entropy  : 0.5999618411064148 ± 0.0026517035713923102
# val_loss           : 0.7380370676517487 ± 0.009437874482978338
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin propagators/ppr/low-rank.gin --bindings='
preprocess_train = False
rank = 256
'
# Completed 10 trials
# test_acc           : 0.7828999996185303 ± 0.004253227092543651
# test_cross_entropy : 0.5959616541862488 ± 0.0042842374605086275
# test_loss          : 0.7507621228694916 ± 0.008108166851350488
# val_acc            : 0.8068000018596649 ± 0.004118250560462895
# val_cross_entropy  : 0.5403760433197021 ± 0.002832358092797155
# val_loss           : 0.6951765179634094 ± 0.005332051400083695
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/pubmed.gin propagators/ppr/low-rank.gin --bindings='
preprocess_train = False
rank = 512
'
# Completed 10 trials
# test_acc           : 0.7960999965667724 ± 0.004437336938623209
# test_cross_entropy : 0.5821191668510437 ± 0.00797269241492792
# test_loss          : 0.760474807024002 ± 0.013552931450735292
# val_acc            : 0.8151999950408936 ± 0.0045782075988075105
# val_cross_entropy  : 0.5339264094829559 ± 0.004410936442240983
# val_loss           : 0.7122820556163788 ± 0.00960455573584395
```
