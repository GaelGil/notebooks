import grain
from jax import numpy as jnp


class Source(grain.DataLoader):
    def __init__(self, src_path: str, target_path: str, pad_id: int):
        self.src = jnp.load(src_path)
        self.target = jnp.load(target_path)
        self.pad_id = pad_id

    def __len__(self):
        return len(self.src)

    def make_padding_mask(self, padded_ids):
        return padded_ids != self.pad_id

    def make_src_mask(self, encoder_input_ids):
        return self.make_padding_mask(encoder_input_ids)

    def make_causal_mask(self, tgt_len: int):
        """
        Returns a boolean mask shape (1, 1, tgt_len, tgt_len), True for allowed (i <= j).
        Returns lower-triangular boolean mask shape (1, 1, tgt_len, tgt_len),
        True for allowed (i >= j).
        """
        tri = jnp.tril(jnp.ones((tgt_len, tgt_len), dtype=jnp.bool_))
        return tri[None, None, :, :]  # (1,1,T,T)

    def make_decoder_self_mask(self, decoder_input_ids):
        T = decoder_input_ids.shape[0]
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        key_nonpad = self.make_padding_mask(decoder_input_ids)
        return causal & key_nonpad[None, :]

    def make_encoder_decoder_mask(self, encoder_input_ids, decoder_input_ids):
        src_len = encoder_input_ids.shape[0]
        tgt_len = decoder_input_ids.shape[0]

        enc_nonpad = self.make_padding_mask(encoder_input_ids)  # (S,)

        # (1, T, S)
        return jnp.broadcast_to(
            enc_nonpad[None, None, :],
            (1, tgt_len, src_len),
        )

    def __getitem__(self, idx):
        encoder_input = self.src[idx]  # encoder input ids padded
        decoder_input = self.target[idx]  # decoder input ids padded
        # print(encoder_input.shape)

        labels = decoder_input[1:]  # the labels are the decoder input shifted by one
        labels_mask = self.make_padding_mask(labels)
        decoder_input = decoder_input[:-1]

        encoder_padding_mask = self.make_padding_mask(
            encoder_input
        )  # mask for encoder to ignore padded tokens
        decoder_self_attention_mask = self.make_decoder_self_mask(
            decoder_input
        )  # mask for decoder self-attention to ignore future tokens

        encoder_decoder_mask = self.make_encoder_decoder_mask(
            encoder_input, decoder_input
        )

        return (
            encoder_input,
            decoder_input,
            labels,
            labels_mask,
            encoder_padding_mask,
            decoder_self_attention_mask,
            encoder_decoder_mask,
        )
