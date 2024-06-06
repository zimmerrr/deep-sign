import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DeepSignConfigV5:
    input_size: int = (
        (33 * 4 + 28) + (21 * 3 + 15 + 6 * 3) + (21 * 3 + 15 + 6 * 3)
    )  # unnormalized input w/ directions

    label_lstm_size: int = 96
    label_lstm_layers: int = 2

    feature_lstm_size: int = 32
    feature_lstm_layers: int = 2

    label_linear_size: int = 96
    num_label: int = 144

    handshape_linear_size: int = 32
    num_handshape: int = 27

    orientation_linear_size: int = 32
    num_orientation: int = 8

    movement_linear_size: int = 32
    num_movement: int = 11

    location_linear_size: int = 32
    num_location: int = 5

    hands_linear_size: int = 8
    num_hands: int = 2

    dropout: float = 0.2
    bidirectional: bool = False
    label_smoothing: float = 0.0


class DeepSignV5(nn.Module):
    def __init__(self, config: DeepSignConfigV5):
        super(DeepSignV5, self).__init__()
        self.config = config

        # RNN Layer
        self.label_rnn = nn.GRU(
            config.input_size,
            config.label_lstm_size,
            config.label_lstm_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        self.features_rnn = nn.GRU(
            config.input_size,
            config.feature_lstm_size,
            config.feature_lstm_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        label_rnn_output_size = (
            config.label_lstm_size * 2
            if config.bidirectional
            else config.label_lstm_size
        )

        feature_rnn_output_size = (
            config.feature_lstm_size * 2
            if config.bidirectional
            else config.feature_lstm_size
        )

        # Handshape Layer
        self.handshape_linear_dropout = nn.Dropout(config.dropout)
        self.handshape_linear = nn.Linear(
            feature_rnn_output_size, config.handshape_linear_size
        )
        self.handshape_dropout = nn.Dropout(config.dropout)
        self.handshape_output = nn.Linear(
            config.handshape_linear_size, config.num_handshape
        )

        # Orientation Layer
        self.orientation_linear_dropout = nn.Dropout(config.dropout)
        self.orientation_linear = nn.Linear(
            feature_rnn_output_size, config.orientation_linear_size
        )
        self.orientation_dropout = nn.Dropout(config.dropout)
        self.orientation_output = nn.Linear(
            config.orientation_linear_size, config.num_orientation
        )

        # Movement Layer
        self.movement_linear_dropout = nn.Dropout(config.dropout)
        self.movement_linear = nn.Linear(
            feature_rnn_output_size, config.movement_linear_size
        )
        self.movement_dropout = nn.Dropout(config.dropout)
        self.movement_output = nn.Linear(
            config.movement_linear_size, config.num_movement
        )

        # Location Layer
        self.location_linear_dropout = nn.Dropout(config.dropout)
        self.location_linear = nn.Linear(
            feature_rnn_output_size, config.location_linear_size
        )
        self.location_dropout = nn.Dropout(config.dropout)
        self.location_output = nn.Linear(
            config.location_linear_size, config.num_location
        )

        # Hands Layer
        self.hands_linear_dropout = nn.Dropout(config.dropout)
        self.hands_linear = nn.Linear(feature_rnn_output_size, config.hands_linear_size)
        self.hands_dropout = nn.Dropout(config.dropout)
        self.hands_output = nn.Linear(config.hands_linear_size, config.num_hands)

        # Label Layer
        feature_output_size = (
            config.num_handshape
            + config.num_orientation
            + config.num_movement
            + config.num_location
            + config.num_hands
        )
        self.label_linear_dropout = nn.Dropout(config.dropout)
        self.label_linear = nn.Linear(
            label_rnn_output_size + feature_output_size,
            config.label_linear_size,
        )
        self.label_dropout = nn.Dropout(config.dropout)
        self.label_output = nn.Linear(config.label_linear_size, config.num_label)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def forward(
        self,
        input,
        target_label=None,
        target_handshape=None,
        target_orientation=None,
        target_movement=None,
        target_location=None,
        target_hands=None,
    ):
        features_output, _ = self.features_rnn(input)
        features_output = features_output[:, -1, :]

        handshape_output = self.handshape_linear_dropout(features_output)
        handshape_output = self.handshape_linear(handshape_output)
        handshape_output = self.handshape_dropout(handshape_output)
        handshape_output = self.handshape_output(handshape_output)

        orientation_output = self.orientation_linear_dropout(features_output)
        orientation_output = self.orientation_linear(orientation_output)
        orientation_output = self.orientation_dropout(orientation_output)
        orientation_output = self.orientation_output(orientation_output)

        movement_output = self.movement_linear_dropout(features_output)
        movement_output = self.movement_linear(movement_output)
        movement_output = self.movement_dropout(movement_output)
        movement_output = self.movement_output(movement_output)

        location_output = self.location_linear_dropout(features_output)
        location_output = self.location_linear(location_output)
        location_output = self.location_dropout(location_output)
        location_output = self.location_output(location_output)

        hands_output = self.hands_linear_dropout(features_output)
        hands_output = self.hands_linear(hands_output)
        hands_output = self.hands_dropout(hands_output)
        hands_output = self.hands_output(hands_output)

        label_output, _ = self.label_rnn(input)
        label_output = label_output[:, -1, :]
        label_output = torch.cat(
            [
                label_output,
                handshape_output,
                orientation_output,
                movement_output,
                location_output,
                hands_output,
            ],
            dim=-1,
        )
        label_output = self.label_linear_dropout(label_output)
        label_output = self.label_linear(label_output)
        label_output = self.label_dropout(label_output)
        label_output = self.label_output(label_output)

        loss = None
        loss_indv = None
        if target_label is not None:
            handshape_loss = self.criterion(
                handshape_output.view(-1, self.config.num_handshape),
                target_handshape.view(-1),
            )
            orientation_loss = self.criterion(
                orientation_output.view(-1, self.config.num_orientation),
                target_orientation.view(-1),
            )
            movement_loss = self.criterion(
                movement_output.view(-1, self.config.num_movement),
                target_movement.view(-1),
            )
            location_loss = self.criterion(
                location_output.view(-1, self.config.num_location),
                target_location.view(-1),
            )
            hands_loss = self.criterion(
                hands_output.view(-1, self.config.num_hands),
                target_hands.view(-1),
            )
            label_loss = self.criterion(
                label_output.view(-1, self.config.num_label),
                target_label.view(-1),
            )

            loss_indv = dict(
                label=label_loss,
                handshape=handshape_loss,
                orientation=orientation_loss,
                movement=movement_loss,
                location=location_loss,
                hands=hands_loss,
            )
            loss = sum(loss_indv.values())

        return label_output, loss, loss_indv

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DeepSignV5()
    input = torch.randn((30, 1662))
    input = F.normalize(input)
    output = model(input)
    output = F.softmax(output)
    # output = F.argmax(output)
    print(output.argmax(-1))
