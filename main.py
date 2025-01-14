import gzip
import pickle
from main_utils import model_runner

def runner():
    EPOCHS = 100
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.001
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    MODEL_ARCHITECTURE = [IMAGE_HEIGHT * IMAGE_WIDTH, 1000, 1000, 1000, 1000, 1000, 10]

    # Load MNIST-Data into memory
    with gzip.open('./mnist-data/mnist.pkl.gz', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    # Validate data shapes 
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT * IMAGE_WIDTH

    model_runner(train_images, train_labels, test_images, test_labels, MODEL_ARCHITECTURE, BATCH_SIZE, EPOCHS, LEARNING_RATE)

runner()
