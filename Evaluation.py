import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.special      import softmax
from scipy              import interp
from transformers       import BertForSequenceClassification
from sklearn.metrics    import confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, roc_curve, auc


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

def load_model(label_dict, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertForSequenceClassification.from_pretrained(config["bert-model"][0],
                                        num_labels = len(label_dict),
                                        output_attentions = False,
                                        output_hidden_states = False)

    model.to(device)
    model.load_state_dict(torch.load(config["pre-trained-model"], map_location=torch.device('cpu')))

    return model


def report(model, X_test, y_test, y_pred, label_dict):
    cm = confusion_matrix(y_test, y_pred, np.array(list((label_dict.values()))), normalize='true')

    from sklearn.metrics import accuracy_score, f1_score
    print("ACCURACY: ", accuracy_score(y_test, y_pred))

    print("F1 Score: ", f1_score(y_test, y_pred, average='micro'))

    # plot_confusion_matrix(  model, 
    #                         X_test, 
    #                         y_test,
    #                         labels = list(label_dict.values()),
    #                         display_labels = list(label_dict.keys()),
    #                         normalize = 'all')
    # plt.show()
    plot_confusion_matrix_show(cm, label_dict.keys())
    
    cr = classification_report(y_test, y_pred, zero_division=True, digits=5, labels=np.array(list((label_dict.values()))), target_names=label_dict.keys())
    print(cr)


    
def evaluate_model(model, dataloader_validation, label_dict):
    _, predictions, true_vals, probs = evaluate(model, dataloader_validation)
    y_pred = accuracy_per_class(label_dict, predictions, true_vals)
    report(model, dataloader_validation, true_vals, y_pred, label_dict)

    # roc_plot_show(5, true_vals, probs)
    

def roc_plot_show(n_classes, y_test, pred_prob):
    # roc curve for classes
    fpr = {}
    tpr = {}
    roc_auc ={}

    for i in range(n_classes):    
        fpr[i], tpr[i], _ = roc_curve(y_test, pred_prob[:,i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /=  n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC curve
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle='-', linewidth=2)

    from itertools import cycle
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix_show(cm, target_names, normalize=True):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('viridis')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
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
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()