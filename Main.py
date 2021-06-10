import Initialization
import Training
import Testing
import Evaluation

def main():
    # train()
    testTL()

# Function to train models
def train():
    df, label_dict, config = Initialization.initialize()
    bert = config["bert"]

    Training.initialize(df)
    dataset_train, dataset_val = Training.encode_data(bert, df, config)
    training_model = Training.setup_bert_model(bert, label_dict, config)

    dataloader_train, dataloader_validation = Training.create_dataloaders(dataset_train, dataset_val, config)
    
    # Training.train(training_model, dataloader_train, dataloader_validation, config)

    loaded_model = Initialization.load_pretrained_model(bert, label_dict, config)
    Evaluation.evaluate_model(loaded_model, dataloader_validation, label_dict)

# Function to test models
def testTL():
    df, label_dict, config = Initialization.initialize()
    bert = config["bert"]

    Testing.initialize(df)
    dataset_val = Testing.encode_data(bert, df, config)
    dataloader_validation = Testing.create_dataloaders(dataset_val, config)

    loaded_model = Initialization.load_pretrained_model(bert, label_dict, config)
    Evaluation.evaluate_model(loaded_model, dataloader_validation, label_dict)

if __name__ == '__main__':
    main()
    