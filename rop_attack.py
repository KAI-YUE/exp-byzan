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

    # obtain the attacker data loader
    attacker_data_loader = fetch_attacker_dataloader(config, dataset.dst_train, attacker_ids, user_data_mapping)
    # attacker_data_loader = test_loader

    # distance measure for losses 
    # dist_metric = metric_registry[config.dist_metric]
    dist_metrics = []
    for key, value in metric_registry.items():
        dist_metrics.append(value)

    # initialize the byzantine model
    ByzantineUpdater = init_attacker(config)

    global_updater = GlobalUpdater(config)
    fedavg_oracle = BenignFedAvg(num_users=config.total_users - config.num_attackers)
    
    # loss_mat = np.zeros((config.total_users-config.num_attackers, config.rounds))
    loss_mat = []
    for i in range(config.total_users-config.num_attackers):
        loss_mat.append([])

    best_testacc = 0.
    heterogeneities_mean, heterogeneities_std = [[] for i in range(len(dist_metrics))], [[] for i in range(len(dist_metrics))] 

    # ref = None
    for comm_round in range(start_round, config.rounds):
        benign_packages = {}
        for i, user_id in enumerate(user_ids):
            updater = LocalUpdater(config, model)
            updater.init_local_dataset(dataset, user_data_mapping[user_id])
            local_loss, local_acc = updater.local_step(criterion)
            
            # loss_mat[i, comm_round] = local_loss
            loss_mat[i].extend(local_loss)
            # loss_mat[i].extend(local_acc)

            local_package = updater.uplink_transmit()
            benign_packages[user_id] = local_package

        heterogeneities = [[] for i in range(len(dist_metrics))]
        loss_mat_arr = np.asarray(loss_mat)
        # after the training, we are able to calculate the empirical heterogeneity over loss
        for ii in range(loss_mat_arr.shape[0]):
            for jj in range(ii+1, loss_mat_arr.shape[0]):
                for kk in range(len(dist_metrics)):
                    heterogeneities[kk].append(dist_metrics[kk](loss_mat_arr[ii], loss_mat_arr[jj]))

        for l, dist_metric in enumerate(metric_registry.keys()):
            mean_hetero = np.mean(heterogeneities[l])
            std_hetero = np.std(heterogeneities[l])
            median_hetero = np.median(heterogeneities[l])
            variation = std_hetero/mean_hetero
            
            heterogeneities_mean[l].append(mean_hetero)
            heterogeneities_std[l].append(std_hetero)
            
            logger.info("{:s}\t\t {:.3f}\t {:.3f}\t {:.3f} \t {:.3f}".format(dist_metric, median_hetero, mean_hetero, std_hetero, variation))

        # =============================================
        # attack process 
        oracle = fedavg_oracle(benign_packages)
        attacker_packages = {}
        reference_attacker, traj = True, None
        powerful = True
        # powerful = False

        indices = []
        for i, attacker_id in enumerate(attacker_ids[:3]):
            indices.extend(user_data_mapping[attacker_id])
        # for i, attacker_id in enumerate(user_ids[:1]):
        #     indices.extend(user_data_mapping[attacker_id])


        # random_attacker_idx = np.random.randint(0, len(attacker_ids), 1)[0]
        # indices = user_data_mapping[attacker_ids[random_attacker_idx]]
        # random_attacker_idx = np.random.randint(0, len(user_ids), 1)[0]
        # indices = user_data_mapping[user_ids[random_attacker_idx]]
        
        # indices = user_data_mapping[attacker_ids[0]]

        for i, attacker_id in enumerate(attacker_ids):
            # powerful = True if comm_round%2 == 0 else False

            # indices = user_data_mapping[attacker_id]

            updater = ByzantineUpdater(config, model)
            updater.init_local_dataset(dataset, indices)
            # traj = updater.local_step(benign_packages=benign_packages, oracle=oracle, network=model, 
            #                    data_loader=attacker_data_loader, criterion=criterion, comm_round=comm_round, 
            #                    momentum=global_updater.momentum, reference_attacker=reference_attacker, 
            #                    attacker_loss_traj=traj, powerful=powerful)
            
            # traj = updater.local_step(benign_packages=benign_packages, oracle=oracle, network=model, 
            #         data_loader=test_loader, criterion=criterion, comm_round=comm_round, 
            #         momentum=global_updater.momentum, reference_attacker=reference_attacker, 
            #         attacker_loss_traj=traj, powerful=powerful)
            
            traj = updater.local_step(benign_packages=benign_packages, oracle=oracle, network=model, 
                    data_loader=updater.data_loader, criterion=criterion, comm_round=comm_round, 
                    momentum=global_updater.momentum, reference_attacker=reference_attacker, 
                    attacker_loss_traj=traj, powerful=powerful)


            powerful = updater.powerful
            # powerful = False

            # reference_attacker = False

            attacker_package = updater.uplink_transmit()
            if updater.complete_attack:
                for j in range(0, len(attacker_ids)):
                    # scaling_factor = np.random.uniform(0.1, 20)
                    scaling_factor = 1
                    attacker_packages[attacker_ids[j]] = attacker_package*scaling_factor
                break

            attacker_packages[attacker_id] = attacker_package

        # clear loss mat in each round
        loss_mat = []
        for i in range(config.total_users-config.num_attackers):
            loss_mat.append([])

        # Update the global model
        global_updater.global_step(model, benign_packages, attacker_packages, record=record)

        # Validate the model performance and log
        best_testacc = validate_and_log(config, model, dummy_train_loader, test_loader, criterion, comm_round, best_testacc, logger, record)

    # heterogeneity = 0.

    # num_elements = loss_mat.shape[0]*(loss_mat.shape[0] - 1)/2
    # heterogeneity /= num_elements

    # logger.info("The empirical {:s} heterogeneity over loss is {:.3f}".format(config.dist_metric, heterogeneity))
    # record["heterogeneity"] = heterogeneity

    # for i, dist_metric in enumerate(metric_registry.keys()):
    #     mean_hetero = np.mean(heterogeneities[i])
    #     std_hetero = np.std(heterogeneities[i])
    #     median_hetero = np.median(heterogeneities[i])
    #     variation = std_hetero/mean_hetero
    #     logger.info("{:s}\t\t {:.3f}\t {:.3f}\t {:.3f} \t {:.3f}".format(dist_metric, median_hetero, mean_hetero, std_hetero, variation))

    heterogeneities_mean = np.asarray(heterogeneities_mean)
    heterogeneities_std = np.asarray(heterogeneities_std)
    record["heterogeneity_mean"] = heterogeneities_mean
    record["heterogeneity_std"] = heterogeneities_std

    for i, dist_metric in enumerate(metric_registry.keys()):
        mean, std = np.mean(heterogeneities_mean[i]), np.mean(heterogeneities_std[i])
        logger.info("{:s}\t {:.3f}\t {:.3f}\t ({:.3f})".format(dist_metric, mean, std, std/mean))

def main(config_path):
    # load the config file, logger, and initialize the output folder
    config = load_config(config_path)
    user_data_mappings = [
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.1/user_dataidx_map_0.10_0.dat",
        # # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.2/user_dataidx_map_0.20_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.3/user_dataidx_map_0.30_0.dat",
        # # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.4/user_dataidx_map_0.40_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.5/user_dataidx_map_0.50_0.dat",
        # # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.6/user_dataidx_map_0.60_0.dat"
        
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/byzantine/a0.1/user_dataidx_map_0.10_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/byzantine/a0.3/user_dataidx_map_0.30_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/byzantine/a0.5/user_dataidx_map_0.50_0.dat",

        "/mnt/ex-ssd/Datasets/user_with_data/fmnist/iid/iid_mapping_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/byzantine/a100/user_dataidx_map_100_1.dat"

        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.01/user_dataidx_map_0.01_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.03/user_dataidx_map_0.03_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.05/user_dataidx_map_0.05_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/a0.1/user_dataidx_map_0.10_0.dat"

        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/k2/user_dataidx_map_2_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/k4/user_dataidx_map_4_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/k6/user_dataidx_map_6_0.dat",
        # "/mnt/ex-ssd/Datasets/user_with_data/fmnist/k8/user_dataidx_map_8_0.dat",
    ]


    attackers = ["rop"]

    # # radius = [0.3]
    aggregators = ["median", "krum", "trimmed_mean" ,"centeredclipping"]
    # aggregators = ["mean"]
    # aggregators = ["median"]

    num_attackers = [2, 6, 10, 14]
    # num_attackers = [10]

    for i, user_data_mapping in enumerate(user_data_mappings):
        for attacker in attackers:
            for aggregator in aggregators:
                for num_att in num_attackers:

                    # config.radius = r
                    config.user_data_mapping = user_data_mapping
                    config.attacker_model = attacker
                    config.aggregator = aggregator

                    config.num_attackers = num_att
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
    config_file = "config/rop.yaml"
    main(config_file)

