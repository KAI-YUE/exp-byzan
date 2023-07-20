import os
import logging
import numpy as np
import pickle
import datetime
import torch

def init_logger(config, output_dir, seed=0):
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
    # initialize data record 
    record["train_acc"] = []
    record["test_acc"] = []
    record["test_loss"] = []

    record["lambda"] = []
    record["coeff"] = []

    return record

