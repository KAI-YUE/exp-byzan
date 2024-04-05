import os
import numpy as np
import pandas as pd

def UCI_HAR(data_path):
    client_index = np.arange(1, 31)
    csv_path = os.path.join(data_path, "UCI_Smartphone_Raw.csv")
    client_data, client_label = load_subjects_data(csv_path, client_index)
    dst_test_set, dst_test_labels = [], []

    # let's split 10% of the data for testing
    # let's do not split for client data, since the user_data_mapping will be used later
    for i, data in enumerate(client_data):
        split = int(0.1 * len(data))
        _, dst_test = data[split:], data[:split]
        _, dst_test_label = client_label[i][split:], client_label[i][:split]
        dst_test_set.append(dst_test)
        dst_test_labels.append(dst_test_label)

    train_dat, train_label = np.vstack(client_data), np.hstack(client_label)
    dst_train = {"images":train_dat, "labels":train_label}

    # since the dataset does not need user id 
    dst_testset = {"images": np.vstack(dst_test_set), 
                "labels": np.hstack(dst_test_labels)}

    properties = {
        "channel": 9,
        "im_size": 128,
        "num_classes": 6,
        "dst_train": dst_train,
        "dst_test": dst_testset
    }

    class dataset_properties: pass
    for key, value in properties.items():
        setattr(dataset_properties, key, value)

    return dataset_properties


def load_subjects_data(csv, list_of_subjects):
    # this functions load data of each subject in list_of_subjects
    # and return the list of subjects data
    data = []
    label = []
    
    df = pd.read_csv(csv)
    
    # turn label into 0 for the first label 
    df.loc[:, "label"] = df["label"].apply(lambda x: x - 1)
    # acc
    df['body_acc_x'] = df['body_acc_x'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_acc_y'] = df['body_acc_y'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_acc_z'] = df['body_acc_z'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    # gyro
    df['body_gyro_x'] = df['body_gyro_x'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_gyro_y'] = df['body_gyro_y'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_gyro_z'] = df['body_gyro_z'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    # total acc
    df['total_acc_x'] = df['total_acc_x'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['total_acc_y'] = df['total_acc_y'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['total_acc_z'] = df['total_acc_z'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    
    for x in list_of_subjects:
        client_pd = df.loc[df['subject'] == x]
        client_label = client_pd["label"].to_numpy() 
        client_data = np.transpose(np.apply_along_axis(np.stack, 1, client_pd.drop(["label","subject"], axis=1).to_numpy()),(0,1,2))
        data.append(client_data)
        label.append(client_label)
    
    return data,label
    