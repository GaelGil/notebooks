import grain

from jax import numpy as jnp
import numpy as np


class Source(grain.DataLoader):
    """
    Source data loader
    """

    def __init__(self, src_path: str, target_path: str, pad_id: int):
        """
        Args:
            src_path: path to the source data
            target_path: path to the target data
            pad_id: pad id

        Returns:
            None
        """
        self.src = np.load(src_path)
        self.target = np.load(target_path)
        self.pad_id = pad_id

    def __len__(self):
        """
        Returns:
            length of the source data
        """
        return len(self.src)

    def make_padding_mask(self, padded_ids):
        """
        Args:
            padded_ids: padded ids

        Returns:
            mask for padded ids (1, seq_len)
        """
        return padded_ids != self.pad_id

    def make_causal_mask(self, tgt_len: int):
        """
        Args:
            tgt_len: target length

        Returns:
            causal mask
        """
        tri = jnp.tril(jnp.ones((tgt_len, tgt_len), dtype=jnp.bool_))
        return tri[None, None, :, :]

    def make_decoder_self_mask(self, decoder_input_ids):
        """
        Args:
            decoder_input_ids: decoder input ids

        Returns:
            mask for decoder self-attention to ignore future tokens
        """
        T = decoder_input_ids.shape[0]
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        key_nonpad = self.make_padding_mask(decoder_input_ids)
        return causal & key_nonpad[None, :]

    def make_encoder_decoder_mask(self, encoder_input_ids, decoder_input_ids):
        """
        Args:
            encoder_input_ids: encoder input ids
            decoder_input_ids: decoder input ids

        Returns:
            mask for encoder-decoder attention to ignore future tokens
        """
        src_len = encoder_input_ids.shape[0]
        tgt_len = decoder_input_ids.shape[0]

        enc_nonpad = self.make_padding_mask(encoder_input_ids)

        return jnp.broadcast_to(
            enc_nonpad[None, None, :],
            (1, tgt_len, src_len),
        )

    def __getitem__(self, idx):
        encoder_input = jnp.asarray(self.src[idx])  # encoder input ids already padded
        decoder_input = jnp.asarray(
            self.target[idx]
        )  # decoder input ids already padded

        labels = decoder_input[1:]  # the labels are the decoder input shifted by one
        labels_mask = self.make_padding_mask(
            labels
        )  # mask for the labels to ignore padded tokens
        decoder_input = decoder_input[:-1]  # remove the last token

        encoder_padding_mask = self.make_padding_mask(
            encoder_input
        )  # mask for encoder to ignore padded tokens
        decoder_self_attention_mask = self.make_decoder_self_mask(
            decoder_input
        )  # mask for decoder self-attention to ignore future tokens

        encoder_decoder_mask = self.make_encoder_decoder_mask(
            encoder_input, decoder_input
        )  # mask for encoder-decoder attention

        return (
            encoder_input,
            decoder_input,
            labels,
            labels_mask,
            encoder_padding_mask,
            decoder_self_attention_mask,
            encoder_decoder_mask,
        )
