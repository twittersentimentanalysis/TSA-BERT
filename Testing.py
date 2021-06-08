
import torch

from torch.utils.data           import TensorDataset, DataLoader, SequentialSampler
from transformers               import BertTokenizer, BertForSequenceClassification


# Initialize dataset for testing
def initialize(df):
    X_val = df.processed_tweet.values
    y_val = df.label.values
    df['data_type'] = 'val'

    return X_val, y_val


# Encode data for testing
def encode_data(df, config):
    tokenizer = BertTokenizer.from_pretrained(config["bert-model"][0], 
                                            do_lower_case = True)

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == 'val'].processed_tweet.values, 
        add_special_tokens = True, 
        return_attention_mask = True, 
        truncation = True,
        padding = True, 
        max_length = config['max-sequence-length'], 
        return_tensors = 'pt'
    )

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].label.values)

    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    return dataset_val


# Set up BERT Multilingual or BETO model
def setup_bert_model(bert, label_dict, config):
    model = BertForSequenceClassification.from_pretrained(config["bert-model"][bert],
                                                        num_labels = len(label_dict),
                                                        output_attentions = False,
                                                        output_hidden_states = False)
    return model


# Create dataloaders
def create_dataloaders(dataset_val, config):
    dataloader_validation = DataLoader( dataset_val, 
                                        sampler = SequentialSampler(dataset_val), 
                                        batch_size = config['batch-size'])
    
    return dataloader_validation


