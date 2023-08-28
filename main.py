import time
import copy
import numpy as np

# PyTorch libraries
import torch

# My libraries
from config import *
from deeplearning.utils import *
from deeplearning.datasets import *
from deeplearning.metric import metric_registry
from fedlearning import *
from fedlearning.aggregators import BenignFedAvg

def federated_learning(config, logger, record):
    """Simulate Federated Learning training process. 
    
    Args:
        config (class):          the configuration.
        logger (logging.logger): a logger for train info output.
        record (dict):           a record for train info saving.  
    """
    torch.random.manual_seed(config.seed), np.random.seed(config.seed)

    # initialize the dataset and dataloader for training and testing 
    dataset = fetch_dataset(config)
    dummy_train_loader = fetch_dataloader(config, dataset.dst_train)
    test_loader = fetch_dataloader(config, dataset.dst_test, shuffle=False)
    model, criterion, user_ids, attacker_ids, user_data_mapping, start_round = init_all(config, dataset, logger)

    # distance measure for losses 
    dist_metric = metric_registry[config.dist_metric]

    # initialize the byzantine model
    ByzantineUpdater = init_attacker(config)

    global_updater = GlobalUpdater(config)
    fedavg_oracle = BenignFedAvg(num_users=config.total_users - config.num_attackers)
    
    # loss_mat = np.zeros((config.total_users-config.num_attackers, config.rounds))
    loss_mat = []
    for i in range(config.total_users-config.num_attackers):
        loss_mat.append([])

    best_testacc = 0.

    # ref = None
    for comm_round in range(start_round, config.rounds):
        benign_packages = {}
        for i, user_id in enumerate(user_ids):
            updater = LocalUpdater(config, model)
            updater.init_local_dataset(dataset, user_data_mapping[user_id])
            local_loss = updater.local_step(criterion)
            
            # loss_mat[i, comm_round] = local_loss
            loss_mat[i].extend(local_loss)

            local_package = updater.uplink_transmit()
            benign_packages[user_id] = local_package

        oracle = fedavg_oracle(benign_packages)
        attacker_packages = {}
        for i, attacker_id in enumerate(attacker_ids):
            updater = ByzantineUpdater(config, model)
            updater.init_local_dataset(dataset, user_data_mapping[user_id])
            updater.local_step(benign_packages=benign_packages, oracle=oracle, network=model,test_loader=test_loader, criterion=criterion, comm_round=comm_round, momentum=global_updater.momentum)

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

    # heterogeneity = 0.
    heterogeneity = []
    loss_mat = np.asarray(loss_mat)
    # after the training, we are able to calculate the empirical heterogeneity over loss
    for i in range(loss_mat.shape[0]):
        for j in range(i+1, loss_mat.shape[0]):
            # heterogeneity += dist_metric(loss_mat[i], loss_mat[j])
            heterogeneity.append(dist_metric(loss_mat[i], loss_mat[j]))

    # num_elements = loss_mat.shape[0]*(loss_mat.shape[0] - 1)/2
    # heterogeneity /= num_elements

    # logger.info("The empirical {:s} heterogeneity over loss is {:.3f}".format(config.dist_metric, heterogeneity))
    # record["heterogeneity"] = heterogeneity

    mean_hetero = np.mean(heterogeneity)
    std_hetero = np.std(heterogeneity)
    print("{:s}, {:.3f}, {:.3f}".format(config.dist_metric, mean_hetero, std_hetero))


def main():
    # load the config file, logger, and initialize the output folder
    config = load_config()
    user_data_mappings = [
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/iid/iid_mapping_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.1/user_dataidx_map_0.10_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.2/user_dataidx_map_0.20_0.dat",
        "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.3/user_dataidx_map_0.30_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.4/user_dataidx_map_0.40_0.dat",
        "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.5/user_dataidx_map_0.50_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.6/user_dataidx_map_0.60_0.dat"
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/k2/user_dataidx_map_2_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/k4/user_dataidx_map_4_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/k6/user_dataidx_map_6_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/k8/user_dataidx_map_8_0.dat",
    ]

    # attackers = ["ipm", "alie"]
    # # radius = [0.3]
    # aggregators = ["median", "krum", "trimmed_mean" ,"centeredclipping"]
    aggregators = ["mean"]
    aggregators = ["median"]
    # num_attackers = [14]

    for i, user_data_mapping in enumerate(user_data_mappings):
    # for attacker in attackers:
        for aggregator in aggregators:
            # for num_att in num_attackers:

            # config.radius = r
            config.user_data_mapping = user_data_mapping
            # config.attacker_model = attacker
            config.aggregator = aggregator

            # config.num_attackers = num_att
            # config.ipm_multiplier = (config.total_users-num_att)/num_att

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
    main()

