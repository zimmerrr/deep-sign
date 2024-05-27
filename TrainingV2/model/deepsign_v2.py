import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DeepSignConfigV2:
    input_size: int = 1662
    num_label: int = 30
    lstm1_size: int = 64
    lstm2_size: int = 128
    linear_size: int = 960


class DeepSignV2(nn.Module):
    def __init__(self, config: DeepSignConfigV2):
        super(DeepSignV2, self).__init__()
        self.lstm1 = nn.LSTM(config.input_size, config.lstm1_size, batch_first=True)
        self.lstm2 = nn.LSTM(config.lstm1_size, config.lstm2_size, batch_first=True)
        self.lstm3 = nn.LSTM(config.lstm2_size, config.lstm1_size, batch_first=True)
        self.linear1 = nn.Linear(config.lstm1_size, config.linear_size)
        self.linear2 = nn.Linear(config.linear_size, config.num_label)

    def forward(self, input):
        output, _ = self.lstm1(input)
        output, _ = self.lstm2(output)
        output, _ = self.lstm3(output)
        # Only pass the last sequence to the linear layer for classification
        output = self.linear1(output[:, -1, :])
        output = self.linear2(output)

        return output


if __name__ == "__main__":
    model = DeepSignV2()
    input = torch.randn((30, 1662))
    input = F.normalize(input)
    output = model(input)
    output = F.softmax(output)
    # output = F.argmax(output)
    print(output.argmax(-1))
