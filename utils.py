import numpy as np

def dataloader(image_arrays, label_arrays, batch_size: int, shuffle: bool):
    num_samples = image_arrays.shape[0]
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield image_arrays[indices[start:end]], label_arrays[indices[start:end]]

def train_model(train_dataloader, model_train_runner, optimizer, criterion):
    for images, labels in train_dataloader:
        outputs = model_train_runner(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def test_model(test_dataloader, model_test_runner):
    for images, labels in test_dataloader:
        pass

def model_runner(train_loader, test_loader, epochs):
    for epoch in range(epochs):
        # TODO: Train model
        # TODO: Test model performance
        pass
