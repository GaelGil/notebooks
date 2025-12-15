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

    def make_causal_mask(self, target_ids):
        """
        Returns lower-triangular boolean mask shape (1, 1, tgt_len, tgt_len),
        True for allowed (i >= j).
        """
        target_len = self.make_padding_mask(padded_ids=target_ids).shape[-1]
        tri = jnp.tril(jnp.ones((target_len, target_len), dtype=jnp.bool_))
        return tri[None, None, :, :]  # will broadcast over batch

    def make_decoder_self_mask(self, decoder_input_ids):
        """
        Produces mask for decoder self-attention combining causal + decoder padding.
        Output shape: (batch, 1, tgt_len, tgt_len), dtype bool.
        semantics: True = allowed (query can attend to key)
        """
        batch, tgt_len = decoder_input_ids.shape
        causal = self.make_causal_mask(tgt_len)  # (1,1,tgt,tgt)
        key_nonpad = self.make_padding_mask(
            decoder_input_ids, self.pad_id
        )  # (batch, tgt)
        key_nonpad = key_nonpad[:, None, None, :]  # (batch,1,1,tgt)
        # Combine: a query position q can attend to key k iff k is not pad AND k<=q (causal)
        return causal & key_nonpad

    def make_encoder_decoder_mask(self, encoder_input_ids, decoder_input_ids):
        """
        Mask for encoder->decoder cross-attention:
        shape (batch, 1, tgt_len, src_len), dtype bool.
        True = encoder token is not padding (decoder may attend to it).
        """
        batch, src_len = encoder_input_ids.shape
        _, tgt_len = decoder_input_ids.shape
        enc_nonpad = self.make_padding_mask(
            encoder_input_ids, self.pad_id
        )  # (batch, src_len)
        enc_nonpad = enc_nonpad[:, None, None, :]  # (batch,1,1,src_len)
        # Broadcast to (batch, 1, tgt_len, src_len)
        return jnp.broadcast_to(enc_nonpad, (batch, 1, tgt_len, src_len))

    def __getitem__(self, idx):
        src_ids_padded = self.src[idx]
        target_ids_padded = self.target[idx]

        src_mask = self.make_padding_mask(src_ids_padded)
        target_mask = self.make_decoder_self_mask(target_ids_padded)
        enc_dec_mask = self.make_encoder_decoder_mask(src_ids_padded, target_ids_padded)

        return src_ids_padded, target_ids_padded, src_mask, target_mask, enc_dec_mask
