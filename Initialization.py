import json
from pandas.core.indexing import is_label_like
import torch
import pandas as pd
from transformers import BertForSequenceClassification

def read_config_file(file):
    # read configuration file
    js = open(file).read()
    config = json.loads(js)

    return config

def initialize_api():
    config = read_config_file('config-api.json')
    label_dict = config["label-dict"]

    return config, label_dict


def initialize():
    config = read_config_file('config.json')
    
    # read csv and add names to columns
    df = pd.read_csv(config['csv-file'],
                        dtype = str,
                        header = 0, 
                        index_col = 'id',
                        usecols = ['id', 'processed_tweet', 'emotion'])
    #df.set_index('id', inplace = True)

    print(df.head())
    print(df.emotion.value_counts())

    # remove empty processed tweets
    df = df[~df.processed_tweet.isnull()]

    print(df.emotion.value_counts())
    print("TOTAL :", len(df))

    # # BALANCE DATA
    # # Shuffle the Dataset.
    # shuffled_df = df.sample(frac=1,random_state=4)

    # # Randomly select 492 observations from the non-fraud (majority class)
    # happy_df = shuffled_df.loc[shuffled_df['emotion'] == 'happy'].sample(n=50,random_state=63)
    # sad_df = shuffled_df.loc[shuffled_df['emotion'] == 'sad'].sample(n=50,random_state=32)
    # # surprise_df = shuffled_df.loc[shuffled_df['emotion'] == 'surprise'].sample(n=315,random_state=42)
    # # nrelevant_df = shuffled_df.loc[shuffled_df['emotion'] == 'not-relevant'].sample(n=45,random_state=42)
    # angry_df = shuffled_df.loc[shuffled_df['emotion'] == 'angry'].sample(n=50,random_state=75)
    # surprise_df = shuffled_df.loc[shuffled_df['emotion'] == 'surprise'].sample(n=17,random_state=22)

    # # Concatenate both dataframes again
    # df = pd.concat([happy_df, sad_df, angry_df, surprise_df])

    import seaborn as sns
    import matplotlib.pyplot as plt
    # #plot the dataset after the undersampling
    plt.figure(figsize=(8, 8))
    sns.countplot('emotion', data=df)
    # plt.title('Balanced Classes')
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



def load_pretrained_model(bert, label_dict, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertForSequenceClassification.from_pretrained(config["bert-model"][bert],
                                        num_labels = len(label_dict),
                                        output_attentions = False,
                                        output_hidden_states = False)

    model.to(device)
    model.load_state_dict(torch.load(config["model"][bert], map_location=torch.device('cpu')))

    return model


def load_model(bert):
    config, label_dict = initialize_api()
    loaded_model = load_pretrained_model(bert, label_dict, config)
    return loaded_model, config, label_dict