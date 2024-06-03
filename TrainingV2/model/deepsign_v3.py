import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DeepSignConfigV3:
    input_size: int = 1662
    num_label: int = 30
    lstm_layers: int = 3
    lstm_size: int = 64
    linear_size: int = 960
    dropout = 0.2


class DeepSignV3(nn.Module):
    def __init__(self, config: DeepSignConfigV3):
        super(DeepSignV3, self).__init__()
        self.config = config
        self.rnn = nn.GRU(
            config.input_size,
            config.lstm_size,
            config.lstm_layers,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(config.dropout)
        self.linear1 = nn.Linear(
            config.lstm_size,
            config.linear_size,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(
            config.linear_size,
            config.num_label,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target=None):
        output, _ = self.rnn(input)
        output = self.lstm_dropout(output[:, -1:, :])
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.linear2(output)

        loss = None
        if target is not None:
            loss = self.criterion(
                output.view(-1, self.config.num_label),
                target[:, -1].view(-1),
            )

        return output, loss

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DeepSignV3()
    input = torch.randn((30, 1662))
    input = F.normalize(input)
    output = model(input)
    output = F.softmax(output)
    # output = F.argmax(output)
    print(output.argmax(-1))
