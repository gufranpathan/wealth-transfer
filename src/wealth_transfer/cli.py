from argparse import ArgumentParser
from .data_prep import DataLoader
from .model_config import model_hyperparams
from .train import Trainer

parser = ArgumentParser(prog='Wealth Transfer')
parser.add_argument('-d','--dhs_path', default="data/dhs_final_labels.csv",help="Path of the dhs_final_labels.csv")
parser.add_argument('-c','--countries_train', default="IA|NP|PK",help="Countries to train the data on")
parser.add_argument('-s','--si_data_path', default="data/si",help="Path of Satellite Images Data")





def main(args=None):
    args = parser.parse_args(args=args)
    data_loader = DataLoader(dhs_path=args.dhs_path,countries_train=args.countries_train,data_dir=args.si_data_path)
    data_loader.run()

    for model_param in model_hyperparams:
        trainer = Trainer(data_loader,model_param)
        trainer.run()
