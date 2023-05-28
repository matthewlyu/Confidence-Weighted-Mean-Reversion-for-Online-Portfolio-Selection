import pandas as pd
from universal import tools
from universal import algos
import numpy as np
import argparse
import warnings

warnings.filterwarnings("ignore")

# set default parameters
parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
parser.add_argument('--fees', type=float, default=0.0003)
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--model', type=str, default='cwmr')
parser.add_argument('--eps', type=int, default=-0.5)

args = parser.parse_args()

model_parser = {
    'cwmr': algos.CWMR(),
}

# load data using tools module
data = tools.dataset(args.data)
print('running:', 'model:', args.model, 'data:', args.data, 'fees:', args.fees)
model = model_parser[args.model]

# Compute performances
index = data.columns
fee = pd.Series(args.fees * np.ones(data.shape[1]), index=index)
result_classic = model.run(data)
result_classic.fee = fee
print(result_classic.final_wealth)
print(result_classic.summary())
