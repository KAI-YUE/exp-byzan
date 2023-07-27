import time

# PyTorch libraries
import torch

# My libraries
from config import *
from deeplearning.utils import *
from deeplearning.datasets import *
from fedlearning import *
from fedlearning.aggregators import BenignFedAvg

def federated_learning(config, logger, record):
    """Simulate Federated Learning training process. 
    
    Args:
        config (class):          the configuration.
        logger (logging.logger): a logger for train info output.
        record (dict):           a record for train info saving.  
    """

    # initialize the dataset and dataloader for training and testing 
    dataset = fetch_dataset(config)
    dummy_train_loader = fetch_dataloader(config, dataset.dst_train)
    test_loader = fetch_dataloader(config, dataset.dst_test)
    model, criterion, user_ids, attacker_ids, user_data_mapping, start_round = init_all(config, dataset, logger)

    # initialize the byzantine model
    ByzantineUpdater = init_attacker(config)

    global_updater = GlobalUpdater(config)
    fedavg_oracle = BenignFedAvg(num_users=config.total_users - config.num_attackers)
    best_testacc = 0.

    for comm_round in range(start_round, config.rounds):
        benign_packages = {}
        for i, user_id in enumerate(user_ids):
            updater = LocalUpdater(config, model)
            updater.init_local_dataset(dataset, user_data_mapping[user_id])
            updater.local_step(criterion)

            local_package = updater.uplink_transmit()
            benign_packages[user_id] = local_package

        oracle = fedavg_oracle(benign_packages)
        attacker_packages = {}
        for i, attacker_id in enumerate(attacker_ids):
            updater = ByzantineUpdater(config, model)
            updater.init_local_dataset(dataset, user_data_mapping[user_id])
            updater.local_step(oracle=oracle, network=model,test_loader=test_loader, criterion=criterion)

            attacker_package = updater.uplink_transmit()
            if updater.complete_attack:
                for j in range(0, len(attacker_ids)):
                    attacker_packages[attacker_ids[j]] = attacker_package
                break

            attacker_packages[attacker_id] = attacker_package

        # Update the global model
        global_updater.global_step(model, benign_packages, attacker_packages, record=record)

        # Validate the model performance and log
        best_testacc = validate_and_log(config, model, dummy_train_loader, test_loader, criterion, comm_round, best_testacc, logger, record)

def main(config_path):
    # load the config file, logger, and initialize the output folder
    config = load_config(config_path)
    
    
    num_attackers = [2, 6, 10, 14]
    # num_attackers = [2]

    # for attacker in attackers:
    #     for aggregator in aggregators:
    for num_att in num_attackers:
                
        # output_dir = init_outputfolder(config)
        # logger = init_logger(config, output_dir, config.seed, attach=attach)

        # config.user_data_mapping = user_data_mapping
        # config.attacker_model = attacker
        # config.aggregator = aggregator
        config.num_attackers = num_att
        
        output_dir = init_outputfolder(config)
        logger = init_logger(config, output_dir, config.seed)

        record = init_record(config)

        if config.device == "cuda":
            torch.backends.cudnn.benchmark = True

        start = time.time()
        federated_learning(config, logger, record)
        end = time.time()

        logger.info("{:.3} mins has elapsed".format((end-start)/60))
        save_record(record, output_dir)

        logger.handlers.clear()

if __name__ == "__main__":
    config_path = "config/signsgd.yaml"
    main(config_path)

