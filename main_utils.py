import numpy as np
from modelv2.mlp import mlp
from model.utils import parameters_init

def dataloader(image_arrays, label_arrays, batch_size: int, shuffle: bool):
    num_samples = image_arrays.shape[0]
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield image_arrays[indices[start:end]], label_arrays[indices[start:end]]

def model_runner(train_images, train_labels, test_images, test_labels, model_size, batch_size, epochs, learning_rate):
    training_phase, testing_phase = mlp(size=(784, 10))

    for epoch in range(1, epochs+1):
        # Create a dataloaders
        trainining_loader = dataloader(train_images, train_labels, batch_size, shuffle=True)
        test_loader = dataloader(test_images, test_labels, batch_size, shuffle=False)
        # Model phases
        average_loss = training_phase(trainining_loader)
        accuracy = testing_phase(test_loader)
        print(f"EPOCH: {epoch} Train_loss: {average_loss} Accuracy: {accuracy}")
