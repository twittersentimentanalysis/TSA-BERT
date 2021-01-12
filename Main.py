import Initialization
import Training
import Evaluation

def main():
    df, label_dict, config = Initialization.initialize()

    X_train, X_val, y_train, y_val = Training.initialize(df)
    dataset_train, dataset_val = Training.encode_data(df, config)
    training_model = Training.setup_bert_model(label_dict, config)

    dataloader_train, dataloader_validation = Training.create_dataloaders(dataset_train, dataset_val, config)
    
    # Training.train(training_model, dataloader_train, dataloader_validation, config)

    loaded_model = Evaluation.load_model(label_dict, config)
    Evaluation.evaluate_model(loaded_model, dataloader_validation, label_dict)

def load_model():
    df, label_dict, config = Initialization.initialize()
    loaded_model = Evaluation.load_model(label_dict, config)
    return loaded_model, config, label_dict

if __name__ == '__main__':
    main()
    