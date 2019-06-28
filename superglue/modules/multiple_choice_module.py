import torch
from torch import nn


class MultipleChoiceModule(nn.Module):
    def __init__(self, n_choices=2):
        super().__init__()

        self.n_choices = n_choices

    def forward(self, immediate_output_dict):
        token_ids = torch.cat(
            [
                immediate_output_dict["_input_"][f"token{i+1}_ids"]
                for i in range(self.n_choices)
            ],
            dim=-1,
        )
        token_masks = torch.cat(
            [
                immediate_output_dict["_input_"][f"token{i+1}_masks"]
                for i in range(self.n_choices)
            ],
            dim=-1,
        )
        token_segments = torch.cat(
            [
                immediate_output_dict["_input_"][f"token{i+1}_segments"]
                for i in range(self.n_choices)
            ],
            dim=-1,
        )

        batch_size, seq_length = token_ids.size()

        token_ids = token_ids.view(batch_size * self.n_choices, -1)
        token_masks = token_masks.view(batch_size * self.n_choices, -1)
        token_segments = token_segments.view(batch_size * self.n_choices, -1)

        return token_ids, token_masks, token_segments
