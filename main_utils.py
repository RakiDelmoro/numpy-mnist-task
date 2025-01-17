import numpy as np

def dataloader(image_arrays, label_arrays, batch_size: int, shuffle: bool, total_samples):
    num_samples = image_arrays.shape[0]
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)
    # indices = indices[:total_samples+1]
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield image_arrays[indices[start:end]], label_arrays[indices[start:end]]

def model_runner(model, train_images, train_labels, test_images, test_labels, model_size, batch_size, epochs, learning_rate):
    training_phase, testing_phase = model

    for epoch in range(1, epochs+1):
        batched_total_samples = train_images.shape[0] // batch_size
        # Create a dataloaders
        trainining_loader = dataloader(train_images, train_labels, batch_size, shuffle=True, total_samples=1000)
        test_loader = dataloader(test_images, test_labels, batch_size, shuffle=False, total_samples=1000)
        # Model phases
        average_loss = training_phase(trainining_loader, learning_rate, batched_total_samples)
        accuracy = testing_phase(test_loader)
        print(f"EPOCH: {epoch} Train_loss: {average_loss} Accuracy: {accuracy}")
