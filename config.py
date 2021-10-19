import argparse
import utils

def hyperparam():
    parser = argparse.ArgumentParser(description="get hyperparameters for training")
    parser.add_argument('root', metavar='root', type=str, help="enter root dir for lane files")
    # parser.add_argument('-lr', '--lr', type=float, required=True, help="learning rate")
    parser.add_argument('-bs', '--batch_size', type=int, required=True, help="batch size")
    parser.add_argument('-e', '--epochs', type=int, required=True, help="# of epochs for training")
    parser.add_argument('-s', '--save_model', type=utils.str2bool, required=True, help="declare whether trained model should be saved or not")
    parser.add_argument('-lm', '--load_model', type=utils.str2bool, required=True, help="load model to continue training")
    args = parser.parse_args()
    return args
    
# py train.py cityscapes_data/cityscapes_data/ -bs 64 -e 5000 -s True -lm True