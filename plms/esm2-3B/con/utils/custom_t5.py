import torch
import torch.nn as nn
from transformers import T5Model, T5Config

class CustomT5Model(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.t5 = T5Model(config)
        self._init_layer_devices()

    def _init_layer_devices(self):
        # Example device assignment (distribute layers across multiple GPUs if available)
        self.layer_devices = []
        for i, layer in enumerate(self.t5.encoder.block):
            device = torch.device(f'cuda:{i % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
            layer.to(device)
            self.layer_devices.append(device)
        for i, layer in enumerate(self.t5.decoder.block):
            device = torch.device(f'cuda:{i % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
            layer.to(device)
            self.layer_devices.append(device)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None):
        # Ensure the input tensor is on the same device as the first encoder layer
        device = self.layer_devices[0] if self.layer_devices else 'cpu'
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(device)

        encoder_outputs = self.t5.encoder(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = encoder_outputs.last_hidden_state

        for i, layer in enumerate(self.t5.decoder.block):
            layer_device = self.layer_devices[len(self.t5.encoder.block) + i]
            sequence_output = sequence_output.to(layer_device)
            sequence_output = layer(
                sequence_output,
                encoder_hidden_states=encoder_outputs.last_hidden_state.to(layer_device),
                encoder_attention_mask=attention_mask.to(layer_device) if attention_mask is not None else None,
                decoder_attention_mask=decoder_attention_mask.to(layer_device) if decoder_attention_mask is not None else None,
            ).last_hidden_state

        return sequence_output