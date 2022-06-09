### Dependencies

Install PyTorch, following the [guidelines](https://pytorch.org/get-started/locally/).
Then install the requirements:

```bash
pip3 install -r requirements.txt
```

#### Optimize PAC-Bayesian Bounds
```bash
python3 real_PACB.py dataset=MNIST num_trials=1 training.gamma=0.01
```

Default configuration are stored in 'config/real.yaml'. 

#### Evaluate margin Bounds
```bash
python3 real_margin.py dataset=MNIST num_trials=1 training.gamma=0.01
```

Default configuration are stored in 'config/grid_search.yaml'. 
