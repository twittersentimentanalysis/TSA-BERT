import json
import torch
import pandas               as pd
import seaborn              as sns
import matplotlib.pyplot    as plt

from transformers           import BertForSequenceClassification

# Function to read configuration file
def read_config_file(file):
    js = open(file).read()
    config = json.loads(js)

    return config

# Function to initialize configuration for REST API
def initialize_api():
    config = read_config_file('config-api.json')
    label_dict = config["label-dict"]

    return config, label_dict

# Function to initialize system to train and test model
def initialize():
    config = read_config_file('config.json')
    
    # read csv and add names to columns
    df = pd.read_csv(config['csv-file'],
                        dtype = str,
                        header = 0, 
                        index_col = 'id',
                        usecols = ['id', 'processed_tweet', 'emotion'])

    print(df.head())
    print(df.emotion.value_counts())

    # remove empty processed tweets
    df = df[~df.processed_tweet.isnull()]

    print(df.emotion.value_counts())
    print("TOTAL :", len(df))
    
    # plot the dataset after the undersampling
    plt.figure(figsize=(8, 8))
    sns.countplot('emotion', data=df)
    plt.show()

    # enumerate categories
    possible_labels = sorted(df.emotion.unique())

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    df['label'] = df.emotion.replace(label_dict)

    print(df.head())
    label_dict = dict(sorted(label_dict.items()))
    print(label_dict)
    
    return df, label_dict, config


# Function to load saved pretrained model
def load_pretrained_model(bert, label_dict, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertForSequenceClassification.from_pretrained(config["bert-model"][bert],
                                        num_labels = len(label_dict),
                                        output_attentions = False,
                                        output_hidden_states = False)

    model.to(device)
    model.load_state_dict(torch.load(config["model"][bert], map_location=torch.device('cpu')))

    return model


# Function to load model for the REST API
def load_model(bert):
    config, label_dict = initialize_api()
    loaded_model = load_pretrained_model(bert, label_dict, config)
    return loaded_model, config, label_dict