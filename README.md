# dna

Implementation of [Towards Distributed Neural Architectures](https://arxiv.org/abs/2506.22389)

```sh
pip install -e .
python train.py --run_name "fineweb-edu"
python sample --run_name "fineweb-edu"
```


```sh
/usr/bin/python3.10 -m venv env
source env/bin/activate
pip install -e .
sbatch run.sh
```