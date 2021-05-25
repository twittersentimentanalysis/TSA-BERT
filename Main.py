import Initialization
import Training
import Testing
import Evaluation

def main():
    # train()
    test()

def train():
    df, label_dict, config = Initialization.initialize()

    Training.initialize(df)
    dataset_train, dataset_val = Training.encode_data(df, config)
    training_model = Training.setup_bert_model(label_dict, config)

    dataloader_train, dataloader_validation = Training.create_dataloaders(dataset_train, dataset_val, config)
    
    # Training.train(training_model, dataloader_train, dataloader_validation, config)

    loaded_model = Evaluation.load_model(label_dict, config)
    Evaluation.evaluate_model(loaded_model, dataloader_validation, label_dict)

def test():
    df, label_dict, config = Initialization.initialize()

    Testing.initialize(df)
    dataset_val = Testing.encode_data(df, config)
    dataloader_validation = Testing.create_dataloaders(dataset_val, config)

    loaded_model = Evaluation.load_model(label_dict, config)
    Evaluation.evaluate_model(loaded_model, dataloader_validation, label_dict)

if __name__ == '__main__':
    main()
    