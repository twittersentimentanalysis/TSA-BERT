import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.special      import softmax
from sklearn.metrics    import confusion_matrix, classification_report

# Function to compute accuracy of each class
def accuracy_per_class(label_dict, preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Emotion: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

    return preds_flat


# Function to evaluate BERT models
def evaluate(model, dataloader_val):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        
        inputs= {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                }

        with torch.no_grad():        
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    # Apply softmax to calculate probabilities
    probs = softmax(predictions, axis=1)

    return loss_val_avg, predictions, true_vals, probs


# Function to report results 
def report(model, X_test, y_test, y_pred, label_dict):
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, np.array(list((label_dict.values()))), normalize='true')
    show_confusion_matrix(cm, label_dict.keys())
    
    # Show performance metrics
    cr = classification_report(y_test, y_pred, zero_division=True, digits=5, labels=np.array(list((label_dict.values()))), target_names=label_dict.keys())
    print(cr)


# Function to evaluate the model in detail
def evaluate_model(model, dataloader_validation, label_dict):
    _, predictions, true_vals, probs = evaluate(model, dataloader_validation)
    y_pred = accuracy_per_class(label_dict, predictions, true_vals)
    report(model, dataloader_validation, true_vals, y_pred, label_dict)


# Function to plot confusion matrix
def show_confusion_matrix(cm, target_names, normalize=True):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('viridis')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()