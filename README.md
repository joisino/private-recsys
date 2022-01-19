# Private Recommender Systems: How Can Users Build Their Own Fair Recommender Systems without Log Data? (SDM 2022)

We consider how a user of a web service can build their own recommender system. Many recommender systems on the Internet are still unfair/undesirable for some users, in which case the users need to leave the service or unwillingly continue to use the system. Our proposed concept, private recommender systems, provides a way for the users to resolve this dilemma.

Paper: https://arxiv.org/abs/2105.12353


## 💿 Dependency

```
$ pip install -r requirements.txt
$ sudo apt install wget unzip
```


## 🗃️ Download and Preprocess Datasets

You can download and preprocess data by the following command. It may take time.

```
$ bash download.sh
```

`hetrec.npy` is the Last.fm dataset. `home_and_kitchen.npy` is the Amazon dataset. `adult_*.npy` and `adult_*.npz` are the Adult dataset.


## 🧪 Evaluation

```
$ python evaluate.py --data 100k --prov cosine --sensitive popularity
$ python evaluate.py --data 100k --prov bpr --sensitive popularity
$ python evaluate.py --data 100k --prov cosine --sensitive old
$ python evaluate.py --data 100k --prov bpr --sensitive old
$ python evaluate.py --data hetrec --prov bpr --sensitive popularity
$ python evaluate.py --data home --prov bpr --sensitive popularity
$ python evaluate_adult.py
```

* `100k` is the MovieLens 100k dataset. `hetrec` is the LastFM dataset. `home` is the Amazon Home and Kitchen dataset.
* `--prov` specifys the algorithm of the service provider's recommender system.
* `--sensitive` specifyies the sensitive attribute. `old` is available only for the MovieLens datasets.

These scripts compute the sums of recalls, NDCGs, least ratios, and entropies for all users. Be sure to divide these values by the number of users to obtain the average values.

When your environment supports multi-processing, run, for example, the following commands to speed up the computation (with background executions):

```
$ python evaluate.py --data 100k --prov cosine --sensitive popularity --split 7 --block 0
$ python evaluate.py --data 100k --prov cosine --sensitive popularity --split 7 --block 1
$ python evaluate.py --data 100k --prov cosine --sensitive popularity --split 7 --block 2
$ python evaluate.py --data 100k --prov cosine --sensitive popularity --split 7 --block 3
$ python evaluate.py --data 100k --prov cosine --sensitive popularity --split 7 --block 4
$ python evaluate.py --data 100k --prov cosine --sensitive popularity --split 7 --block 5
$ python evaluate.py --data 100k --prov cosine --sensitive popularity --split 7 --block 6
$ python summary.py 7
```

## 🖋️ Citation

```
@inproceedings{sato2022retrieving,
  author    = {Ryoma Sato},
  title     = {Private Recommender Systems: How Can Users Build Their Own Fair Recommender Systems without Log Data?},
  booktitle = {Proceedings of the 2022 {SIAM} International Conference on Data Mining, {SDM}},
  year      = {2022},
}
```