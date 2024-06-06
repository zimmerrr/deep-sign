import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DeepSignConfigV4:
    input_size: int = 1662
    num_label: int = 30
    num_category: int = 30
    lstm_layers: int = 3
    lstm_size: int = 64
    linear_size: int = 960
    dropout: float = 0.2
    bidirectional: bool = False
    loss_whole_sequence: bool = False
    label_smoothing: float = 0.0


class DeepSignV4(nn.Module):
    def __init__(self, config: DeepSignConfigV4):
        super(DeepSignV4, self).__init__()
        self.config = config
        self.rnn = nn.GRU(
            config.input_size,
            config.lstm_size,
            config.lstm_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        self.lstm_dropout = nn.Dropout(config.dropout)
        self.linear1 = nn.Linear(
            config.lstm_size * 2 if config.bidirectional else config.lstm_size,
            config.linear_size,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(
            config.linear_size,
            config.num_label,
        )
        self.output_categ = nn.Linear(
            config.linear_size,
            config.num_category,
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def forward(self, input, target=None, target_category=None):
        output, _ = self.rnn(input)
        output = self.lstm_dropout(output)
        output = self.linear1(output)
        output = self.dropout(output)
        output_categ = self.output_categ(output)
        output = self.output(output)

        if not self.config.loss_whole_sequence:
            output = output[:, -1:, :]
            if target is not None:
                target = target[:, -1:]

        loss = None
        if target is not None:
            loss = self.criterion(
                output.view(-1, self.config.num_label),
                target.view(-1),
            )

        if target_category is not None:
            output_categ = output_categ[:, -1:, :]
            target_category = target_category[:, -1:]
            loss += self.criterion(
                output_categ.view(-1, self.config.num_category),
                target_category.view(-1),
            )

        return output, loss

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DeepSignV4()
    input = torch.randn((30, 1662))
    input = F.normalize(input)
    output = model(input)
    output = F.softmax(output)
    # output = F.argmax(output)
    print(output.argmax(-1))