import json
import pandas as pd

def initialize():
    # read configuration file
    js = open('config.json').read()
    config = json.loads(js)

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

    # # BALANCE DATA
    # # Shuffle the Dataset.
    # shuffled_df = df.sample(frac=1,random_state=4)

    # # Randomly select 492 observations from the non-fraud (majority class)
    # happy_df = shuffled_df.loc[shuffled_df['emotion'] == 'happy'].sample(n=2700,random_state=63)
    # sad_df = shuffled_df.loc[shuffled_df['emotion'] == 'sad'].sample(n=2700,random_state=32)
    # # surprise_df = shuffled_df.loc[shuffled_df['emotion'] == 'surprise'].sample(n=315,random_state=42)
    # # nrelevant_df = shuffled_df.loc[shuffled_df['emotion'] == 'not-relevant'].sample(n=45,random_state=42)
    # angry_df = shuffled_df.loc[shuffled_df['emotion'] == 'angry'].sample(n=2700,random_state=75)

    # # Concatenate both dataframes again
    # df = pd.concat([happy_df, sad_df, angry_df])

    import seaborn as sns
    import matplotlib.pyplot as plt
    #plot the dataset after the undersampling
    plt.figure(figsize=(8, 8))
    sns.countplot('emotion', data=df)
    plt.title('Balanced Classes')
    plt.show()

    # enumerate categories
    possible_labels = df.emotion.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    df['label'] = df.emotion.replace(label_dict)

    print(df.head())

    return df, label_dict, config