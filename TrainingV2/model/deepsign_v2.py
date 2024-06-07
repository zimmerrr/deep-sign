import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DeepSignConfigV2:
    input_size: int = 1662
    num_label: int = 30
    lstm_layers: int = 3
    lstm_size: int = 64
    linear_size: int = 960
    dropout = 0.2


class DeepSignV2(nn.Module):
    def __init__(self, config: DeepSignConfigV2):
        super(DeepSignV2, self).__init__()
        self.lstm1 = nn.LSTM(
            config.input_size,
            config.lstm_size,
            config.lstm_layers,
            batch_first=True,
            dropout=config.dropout,
        )
        # self.lstm2 = nn.LSTM(config.lstm1_size, config.lstm2_size, batch_first=True)
        # self.lstm3 = nn.LSTM(config.lstm2_size, config.lstm3_size, batch_first=True)
        self.linear1 = nn.Linear(
            config.lstm_layers * config.lstm_size,
            config.lstm_layers * config.linear_size,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(
            config.lstm_layers * config.linear_size,
            config.num_label,
        )

    def forward(self, input, hn=None, cn=None):
        batch_size = input.shape[0]
        output, (hn, cn) = self.lstm1(input, (hn, cn))
        output = self.linear1(hn.view((batch_size, -1)))
        output = self.dropout(output)
        output = self.linear2(output)

        return output

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DeepSignV2()
    input = torch.randn((30, 1662))
    input = F.normalize(input)
    output = model(input)
    output = F.softmax(output)
    # output = F.argmax(output)
    print(output.argmax(-1))
