import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSign(nn.Module):
    def __init__(self):
        super(DeepSign, self).__init__()
        self.lstm1 = nn.LSTM(1662, 64)
        self.lstm2 = nn.LSTM(64, 128)
        self.lstm3 = nn.LSTM(128, 64)
        self.linear1 = nn.Linear(30 * 64, 960)
        self.linear2 = nn.Linear(960, 30)

    def forward(self, input):
        output, _ = self.lstm1(input)
        output, _ = self.lstm2(output)
        output, _ = self.lstm3(output)
        output = output.view(-1)
        output = self.linear1(output)
        output = self.linear2(output)

        return output
        

if __name__ == "__main__":
    model = DeepSign()
    input = torch.randn((30, 1662))
    input = F.normalize(input)
    output = model(input)
    output = F.softmax(output)
    # output = F.argmax(output)
    print(output.argmax(-1))
