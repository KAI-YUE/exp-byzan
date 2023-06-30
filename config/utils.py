import os
import logging
import numpy as np
import pickle
import datetime

def init_logger(config, output_dir, seed=0, attach=True):
    """Initialize a logger object. 
    """
    log_level = "INFO"
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    fh = logging.FileHandler(os.path.join(output_dir, "main.log"))
    fh.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)

    logger.addHandler(fh)
    if attach:
        logger.addHandler(sh)
    logger.info("-"*80)
    logger.info("Run with seed {:d}.".format(seed))

    attributes = filter(lambda a: not a.startswith('__'), dir(config))
    for attr in attributes:
        logger.info("{:<20}: {}".format(attr, getattr(config, attr)))

    return logger

def parse_dataset_type(config):
    if "fmnist" in config.train_data_dir:
        type_ = "fmnist"
    elif "mnist" in config.train_data_dir:
        type_ = "mnist"
    elif "cifar" in config.train_data_dir:
        type_ = "cifar"
    
    return type_


def init_outputfolder(config):
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    current_time = datetime.datetime.now()
    current_time_str = datetime.datetime.strftime(current_time, '%m%d_%H%M')

    output_dir = os.path.join(config.output_folder, current_time_str)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config.output_dir = output_dir

    return output_dir


def save_record(record, output_dir):
    with open(os.path.join(output_dir, "record.dat"), "wb") as fp:
        pickle.dump(record, fp)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_record(config):
    record = {}
    # put some config info into record
    record["tau"] = config.tau
    record["batch_size"] = config.batch_size
    record["lr"] = config.lr
    record["momentum"] = config.momentum
    record["weight_decay"] = config.weight_decay

    # initialize data record 
    record["train_acc"] = []
    record["test_acc"] = []
    record["test_loss"] = []

    record["lambda"] = []
    record["coeff"] = []

    return record

