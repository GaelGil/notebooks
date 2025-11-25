Traceback (most recent call last):
  File "/home/gg/git_repos/notebooks/transformer/main.py", line 179, in <module>
    main()
    ~~~~^^
  File "/home/gg/git_repos/notebooks/transformer/main.py", line 144, in main
    train(
    ~~~~~^
        state=state,
        ^^^^^^^^^^^^
    ...<5 lines>...
        step=step,
        ^^^^^^^^^^
    )
    ^
  File "/home/gg/git_repos/notebooks/transformer/utils/train_eval.py", line 44, in train
    state, _ = train_step(state=state, batch=batch, dropout_rng=rng)
               ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/transformer/utils/train_eval.py", line 121, in train_step
    loss, grads = grad_fn(state.params)
                  ~~~~~~~^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/transformer/utils/train_eval.py", line 104, in loss_fn
    logits = state.apply_fn(
        {"params": params},
    ...<5 lines>...
        rngs={"dropout": dropout_rng},
    )
  File "/home/gg/git_repos/notebooks/transformer/transformer/model.py", line 526, in __call__
    target_pos = self.src_pe(x=target_embeddings, is_training=is_training)
  File "/home/gg/git_repos/notebooks/transformer/transformer/model.py", line 62, in __call__
    x = x + self.pe
        ~~^~~~~~~~~
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py", line 1141, in op
    return getattr(self.aval, f"_{name}")(self, *args)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py", line 604, in deferring_binary_op
    return binary_op(*args)
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/numpy/ufunc_api.py", line 183, in __call__
    return call(*args)
  File "/home/gg/git_repos/notebooks/transformer/.venv/lib/python3.13/site-packages/jax/_src/numpy/ufuncs.py", line 1238, in add
    out = lax.add(x, y)
TypeError: add got incompatible shapes for broadcasting: (32, 349, 512), (1, 350, 512).
--------------------
For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.