import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TorchRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear_connecting = nn.Linear(hidden_size, hidden_size)
        self.rnn2 = nn.RNN(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

        # Model properties
        self.loss_function = nn.MSELoss()
        # Note: Adam is smarter than SGD for simplicity we will use SGD
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

        # Model parameters
        self.input_to_hidden_w = self.rnn1.weight_ih_l0
        self.hidden_to_hidden_w = self.rnn1.weight_hh_l0
        self.input_to_hidden_b = self.rnn1.bias_ih_l0
        self.hidden_to_hidden_b = self.rnn1.bias_hh_l0
        self.linear_out_w = self.linear_out.weight
        self.linear_out_b = self.linear_out.bias

    def forward(self, x):
        x = x.unsqueeze(-1)
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)
        rnn1_out, _ = self.rnn1(x, h_0)
        ln_to_rnn2 = self.linear_connecting(rnn1_out)
        rnn2_out, _ = self.rnn2(ln_to_rnn2)
        out = self.linear_out(rnn2_out)
        return out

    def test_runner(self, x_train, y_train):
        model_pred = self.forward(x_train)
        loss = self.loss_function(model_pred, y_train.unsqueeze(-1))
        # get gradients
        self.optimizer.zero_grad()
        loss.backward()
        list_gradients = list([params.grad for _, params in self.named_parameters()])
        return loss.item(), list_gradients

    def runner(self, x_train, y_train, x_test, y_test, epochs):
        for epoch in range(epochs):
            output = self.forward(x_train)
            loss = self.loss_function(output, y_train.unsqueeze(-1))
            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Check model performance 10 epochs interval
            if (epoch + 1) % 10 == 0: print(f'Epoch [{epoch+1}/{epoch}], Loss: {loss.item():.4f}')

        # Make predictions
        with torch.no_grad(): predictions = self.forward(x_test).squeeze(2).numpy()

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[0].numpy(), label='True')
        plt.plot(predictions[0], label='Predicted')
        plt.legend()
        plt.savefig('torch_prediction.png')
        plt.show()
