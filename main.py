import os
import pickle
import time
import numpy as np

# PyTorch libraries
import torch

# My libraries
from config import *
from deeplearning.utils import *
from deeplearning.datasets import *
from deeplearning.dataset import *
from fedlearning import *


def federated_learning(config, logger, record):
    """Simulate Federated Learning training process. 
    
    Args:
        config (class):          the configuration.
        logger (logging.logger): a logger for train info output.
        record (dict):           a record for train info saving.  
    """

    dataset = fetch_dataset(config)
    test_loader = fetch_dataloader(config, dataset.dst_test)
    model, criterion, user_ids,  user_data_mapping, start_round = init_all(config, dataset, logger)

    global_updater = GlobalUpdater(config)
    best_testacc = 0.

    for comm_round in range(config.rounds):
        params = []
        local_packages = {}
        for i, user_id in enumerate(user_ids):
            updater = LocalUpdater(config, dataset.dst_train)
            
            updater.local_step(model, config=config)

            local_package = updater.uplink_transmit()
            local_packages[user_id] = local_package

            # global_xs = torch.cat((global_xs, updater.xs))
            params.append(updater.local_weight)

        # Update the global model
        global_updater.global_step(model, local_packages, record=record)

        # Validate the model performance and log
        logger.info("-"*50)
        logger.info("Communication Round {:d}".format(comm_round))
        testacc = test(test_loader, model, criterion, config, logger, record)

        # remember best prec@1 and save checkpoint
        is_best = (testacc > best_testacc)
        if is_best:
            best_testacc = testacc
            save_checkpoint({"comm_round": comm_round + 1,
                "state_dict": model.state_dict()},
                # config.output_dir + "/checkpoint_epoch{:d}.pth".format(epoch))
                config.output_dir + "/checkpoint.pth")

def main():
    # load the config file, logger, and initialize the output folder
    config = load_config()
    output_dir = init_outputfolder(config)
    logger = init_logger(config, output_dir, config.seed)
    
    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True

    start = time.time()
    record = init_record(config)
    federated_learning(config, logger, record)
    save_record(config, record)

    end = time.time()
    logger.info("{:.3} mins has elapsed".format((end-start)/60))

if __name__ == "__main__":
    main()

