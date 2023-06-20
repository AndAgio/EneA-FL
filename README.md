# EneA-FL
Repository containing experiments for Energy Aware serverless Federated Learning (EneA-FL).

### Version 1.0 available

This version contains the code to run a federation containing N workers, using the FedAvg algorithm as the aggregation process.
At each federation round K workers are sampled randomly, local training is performed, along with local model propagation to the server and the global model aggregation.\
Two datasets are supported:
- FEMNIST (image classification task of hand-written digits), with 62 different classes
- SENT140 (sentiment analysis task of tweets), with 2 classes

The datasets are taken from LEAF whose implementation is modified to support federation over a pre-defined number of workers N.
It is also possible to select the maximum amount of samples that a worker can have (using the max_spw parameter). Moreover, 4 different sampling modes are available to distribute the samples over the workers of the federation:
1. iid+sim: this represents the baseline case, where data are distributed amongst workers in a i.i.d. fashion and each worker has a similar amount of number of samples used during training -> (perfectly balanced federation)
2. iid+nsim: this represents the case where data are distributed amongst workers in a i.i.d. fashion, but the number of samples given to each worker can differ a lot -> (imbalance in terms of sample size)
3. niid+sim: this represents the case where data are NOT distributed amongst workers in an i.i.d. fashion (labels distribution varies over workers), and the number of samples given to each worker is similar -> (imbalance in terms of sample distribution)
4. niid+nsim: the case where both sample sizes and their distributions are imbalanced.

#### Code execution

The command to run an instance of federation learning is:
```
python main.py --dataset="sent140" --num_workers=20 --max_spw=100 --sampling_mode='iid+sim' --num_rounds=100 --eval_every=1 --clients_per_round=10 --lr=0.001
```
where the options represent the following:
- dataset: the selected dataset ("femnist" or "sent140")
- num_workers: the number of workers used to build the federation (the dataset is split depending on the num_workers selected)
- max_spw: maximum number of samples that a worker can hold (the dataset is split depending on the max_spw selected)
- sampling_mode: the mode selected to split the data amongst workers (options are: iid+sim, iid+nsim, niid+sim, niid+nsim)
- num_rounds: the number of rounds used to complete the federation process
- eval_every: the number of federation rounds after which the global model is tested on every worker
- clients_per_round: the number of clients randomly selected at each round to execute the training process
- lr: learning rate


The command to run a single execution of training without federation (learning only locally on a single machine) is the following:
```
python train_non_fl.py --dataset="sent140" --epochs=100 --batch_size=10 --lr=0.1
```
where options are the same as above except epochs which defines the number of epochs used to train the model.

#### Available and missing features

This version contains the following features:
- [x] Datasets import
- [x] Dataset federation depending on workers and sampling mode
- [x] FedAvg like federation process
- [ ] Energy measurement module
- [ ] Global energy management module
- [ ] Containerization
- [ ] Local device identification
- [ ] Local energy policy
