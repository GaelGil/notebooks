from vision_transformer.model import VisionTransformer
import jax.numpy as jnp
import jax


def initialize_model(config):
    model: VisionTransformer = VisionTransformer(
        num_classes=config.NUM_CLASSES,
        patch_size=config.PATCH_SIZE,
        d_model=config.D_MODEL,
        N=config.N,
        n_heads=config.H,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        img_size=config.IMG_SIZE,
        in_channels=config.IN_CHANNELS,
        training=False
    )

    rng: jax.random.PRNGKey = jax.random.PRNGKey(0)

    dummy_input = jnp.zeros(
        (config.BATCH_SIZE, config.NUM_PATCHES, config.IN_CHANNELS, config.PATCH_SIZE),
        dtype=jnp.int32,
    )
    dummy_mask = jnp.zeros((config.BATCH_SIZE, config.NUM_PATCHES+1), dtype=jnp.float32)

    # Initialize with dummy inputs
    variables = model.init(
        rng,
        x=dummy_input,
        src_mask=dummy_mask,
    )

    params = variables["params"]

    return model, params

Ah! Now we’re at a classic **argument mismatch** problem in Flax. Let’s unpack it carefully.

---

## 🔹 The error

```text
TypeError: EncoderBlock.__call__() missing 1 required positional argument: 'src_mask'
```

* In your `EncoderBlock` class, `__call__` is defined like:

```python
class EncoderBlock(nn.Module):
    ...
    def __call__(self, x, src_mask):
        ...
```

* Flax’s `nn.Sequential`-style combinators (like `nn.Sequential` or `nn.ModuleList` / `linen/combinators.py`) **call layers with a single positional argument** by default.

In your traceback:

```python
x = self.blocks(x=x, src_mask=mask)
```

* `self.blocks` is likely a `nn.Sequential` / `nn.ModuleList`
* So each layer (`EncoderBlock`) is being called as `layer(outputs)` **without passing `src_mask`**
* That’s why Python complains.

---

## 🔹 How to fix

You have **two main options**:

### 1️⃣ Use `functools.partial` or `lambda` for extra arguments

Wrap the layer so that it always receives `src_mask`:

```python
self.blocks = [
    lambda x, mask=src_mask: EncoderBlock(...)(x, mask)
    for _ in range(num_layers)
]
```

Or, in Flax:

```python
self.blocks = [
    EncoderBlock(..., src_mask=src_mask)  # not directly supported, see note
]
```

⚠️ But Flax modules are immutable, so the usual fix is to **pass extra arguments explicitly in the loop**:

```python
for block in self.blocks:
    x = block(x, src_mask)
```

---

### 2️⃣ Don’t use `nn.Sequential` for blocks that need extra arguments

Instead of:

```python
self.blocks = nn.Sequential([EncoderBlock(...) for _ in range(num_layers)])
```

Do:

```python
self.blocks = [EncoderBlock(...) for _ in range(num_layers)]

for block in self.blocks:
    x = block(x, src_mask)
```

This is the **most common Flax pattern**, because `nn.Sequential` doesn’t handle extra arguments well.

---

✅ **Key takeaway**:

* `nn.Sequential` (or `linen.combinators`) passes **only the first argument** by default.
* If your module needs extra arguments like `src_mask`, call each layer manually in a loop.

---

I can rewrite your `Encoder` / `EncoderBlock` section correctly so `src_mask` is passed to every block — this will likely fix the current error and prevent new ones.

Do you want me to do that?
