import torch
from transformers import BertTokenizer


def get_emotion(text, model, bert, config, label_dict):
    tokenizer = BertTokenizer.from_pretrained(config["bert-model"][bert], 
                                            do_lower_case = True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Encode and evaluate
    encoded_review = tokenizer.encode_plus(
        text,
        max_length = 256, 
        truncation = True,
        add_special_tokens = True,
        return_token_type_ids = False,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    # apply softmax
    softmax = torch.softmax(output[0], dim=1)

    emotions_prob_softmax = {}
    possible_labels = config['label-dict'].keys()
    for index, possible_label in enumerate(possible_labels):
        index = label_dict[possible_label]
        emotions_prob_softmax[possible_label] = round(softmax[0][index].item(), 5)

    emotions = {"emotions": emotions_prob_softmax}

    return emotions
