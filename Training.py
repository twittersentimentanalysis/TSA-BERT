
import torch
import random
import Evaluation
import numpy as np

from tqdm                       import tqdm
from sklearn.metrics            import f1_score, accuracy_score
from sklearn.model_selection    import train_test_split, StratifiedKFold
from torch.utils.data           import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers               import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup



def initialize(df):
    X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                    df.label.values, 
                                                    test_size = 0.2, 
                                                    random_state = 42, 
                                                    stratify = df.label.values)

    df['data_type'] = ['not_set']*df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'
    # print(df.groupby(['category', 'label', 'data_type']).count())

    return X_train, X_val, y_train, y_val


def encode_data(df, config):
    tokenizer = BertTokenizer.from_pretrained(config["bert-model"][0], 
                                            do_lower_case = True)

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == 'train'].processed_tweet.values, 
        add_special_tokens = True, 
        return_attention_mask = True, 
        truncation = True,
        padding = True, 
        max_length = config['max-sequence-length'], 
        return_tensors = 'pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == 'val'].processed_tweet.values, 
        add_special_tokens = True, 
        return_attention_mask = True, 
        truncation = True,
        padding = True, 
        max_length = config['max-sequence-length'], 
        return_tensors = 'pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    print("LENGTH TRAINING: " + str(len(dataset_train)))
    print("LENGTH VALIDATION: " + str(len(dataset_val)))

    return dataset_train, dataset_val



def setup_bert_model(label_dict, config):
    model = BertForSequenceClassification.from_pretrained(config["bert-model"][0],
                                                        num_labels = len(label_dict),
                                                        output_attentions = False,
                                                        output_hidden_states = False)
    return model



def create_dataloaders(dataset_train, dataset_val, config):
    dataloader_train = DataLoader(  dataset_train, 
                                    sampler = RandomSampler(dataset_train), 
                                    batch_size = config['batch-size'])

    dataloader_validation = DataLoader( dataset_val, 
                                        sampler = SequentialSampler(dataset_val), 
                                        batch_size = config['batch-size'])
    
    return dataloader_train, dataloader_validation

def setup_optimizer(model, config):    
    optimizer = AdamW(model.parameters(),
                    lr = config['adam']['lr'], 
                    eps = config['adam']['eps'],
                    weight_decay=config['adam']['lr']/config['epochs'])
    return optimizer

def setup_scheduler(dataloader_train, optimizer, config):
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train)*config['epochs'])
    return scheduler


def score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    # return f1_score(labels_flat, preds_flat, average='weighted')
    return accuracy_score(labels_flat, preds_flat)


def train(model, dataloader_train, dataloader_validation, config):
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(dataloader_train, optimizer, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in tqdm(range(1, config['epochs'] + 1)):
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       
            
            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            
        
        torch.save(model.state_dict(), f'{config["model-path"]}_{epoch}.model')
            
        tqdm.write(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals, _  = Evaluation.evaluate(model, dataloader_validation)
        val_score = score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'Accuracy Score (Weighted): {val_score}')