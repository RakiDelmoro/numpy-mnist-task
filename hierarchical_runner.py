import gzip
import pickle
from hierarchical_model.model import network
from hierarchical_model.utils import dataloader

def runner():
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28

    # Load MNIST-Data into memory
    with gzip.open('./mnist-data/mnist.pkl.gz', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    # Validate data shapes 
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT * IMAGE_WIDTH

    train_model, test_model = network()

    for epoch in range(500):
        training_loader = dataloader(train_images, train_labels, batch_size=2098, shuffle=True)
        validation_loader = dataloader(test_images, test_labels, batch_size=2098, shuffle=True)
        loss_avg = train_model(training_loader)
        accuracy = test_model(validation_loader)
        print(f"EPOCH: {epoch+1} Loss: {loss_avg} Accuracy: {accuracy}")

runner()
