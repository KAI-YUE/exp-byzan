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
    dummy_train_loader = fetch_dataloader(config, dataset.dst_train)
    test_loader = fetch_dataloader(config, dataset.dst_test)
    model, criterion, user_ids,  user_data_mapping, start_round = init_all(config, dataset, logger)

    global_updater = GlobalUpdater(config)
    best_testacc = 0.

    for comm_round in range(start_round, config.rounds):
        local_packages = {}
        for i, user_id in enumerate(user_ids):
            updater = LocalUpdater(config, model, dataset, user_data_mapping[user_id])
            updater.local_step(criterion)

            local_package = updater.uplink_transmit()
            local_packages[user_id] = local_package
            
            # logger.info("User {} has finished local training".format(user_id))
            # validate_and_log(config, updater.local_model, dummy_train_loader, test_loader, criterion, comm_round, best_testacc, logger, record)

        # Update the global model
        global_updater.global_step(model, local_packages, record=record)

        # Validate the model performance and log
        best_testacc = validate_and_log(config, model, dummy_train_loader, test_loader, criterion, comm_round, best_testacc, logger, record)

def main():
    # load the config file, logger, and initialize the output folder
    config = load_config()
    output_dir = init_outputfolder(config)
    logger = init_logger(config, output_dir, config.seed)
    record = init_record(config)

    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True

    start = time.time()
    federated_learning(config, logger, record)
    end = time.time()

    logger.info("{:.3} mins has elapsed".format((end-start)/60))
    save_record(config, record)

if __name__ == "__main__":
    main()

