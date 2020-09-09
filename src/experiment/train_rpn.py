import sys
sys.path.append("/data")
import json
import argparse
from src.experiment import common_functions as cmf
import warnings
import torch
import torch.nn as nn

warnings.simplefilter("ignore", UserWarning)




""" Get parameters """
def _get_argument_params():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--config_path",
        default="src/experiment/options/default.yml", help="Path to config file.")
	parser.add_argument("--method_type",
        default="ensemble", help="Method type among [||].")
	parser.add_argument("--dataset",
        default="charades_rpn", help="Dataset to train models [|].")
	parser.add_argument("--num_workers", type=int,
        default=4, help="The number of workers for data loader.")
	parser.add_argument("--debug_mode" , action="store_true", default=False,
		help="Train the model in debug mode.")

	params = vars(parser.parse_args())
	print(json.dumps(params, indent=4))
	return params

""" Training the network """
def train(config):

    """ Prepare data loader and model"""
    dsets, L = cmf.get_loader(dataset, split=["train", "test"],
                              loader_configs=[config["train_loader"], config["test_loader"]],
                              num_workers=config["misc"]["num_workers"])

    # Prepare tensorboard

    """ Run training network """
    # load config values
    eval_every = config["evaluation"].get("every_eval", 1) # epoch
    print_every = config["misc"].get("print_every", 1) # iteration
    num_step = config["optimize"].get("num_step", 30) # epoch
    # We evaluate initialized model
    print("=====> # of iteration per one epoch: {}".format(len(L["train"])))

    for epoch in range(num_step):
        # training loop
        for batch in L["train"]:
            pass

        # validate current model
        if (epoch > eval_after) and (epoch % eval_every == 0):
            pass



if __name__ == "__main__":
    # get parameters from cmd
    params = _get_argument_params()
    global M, dataset
    M, dataset, config = cmf.prepare_experiment(params)

    # train network
    train(config)
