import gzip
import pickle
from model.mlp import model
from modelv3.mlp import model
from main_utils import model_runner

def runner():
    EPOCHS = 100
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    MODEL_ARCHITECTURE = [IMAGE_HEIGHT * IMAGE_WIDTH, 1420, 1420, 10]
    
    MODEL = model(MODEL_ARCHITECTURE)
    # MODEL = model(MODEL_ARCHITECTURE)
    
    # Load MNIST-Data into memory
    with gzip.open('./mnist-data/mnist.pkl.gz', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    # Validate data shapes 
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT * IMAGE_WIDTH

    model_runner(MODEL, train_images, train_labels, test_images, test_labels, MODEL_ARCHITECTURE, BATCH_SIZE, EPOCHS, LEARNING_RATE)

runner()
