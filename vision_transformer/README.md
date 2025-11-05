Traceback (most recent call last):
  File "/home/gg/git_repos/notebooks/vision_transformer/main.py", line 100, in <module>
    main()
    ~~~~^^
  File "/home/gg/git_repos/notebooks/vision_transformer/main.py", line 84, in main
    train(
    ~~~~~^
        state=state,
        ^^^^^^^^^^^^
    ...<4 lines>...
        logger=logging,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/home/gg/git_repos/notebooks/vision_transformer/utils/train_eval.py", line 69, in train
    manager.save(
    ~~~~~~~~~~~~^
        step=epoch,
        ^^^^^^^^^^^
    ...<3 lines>...
        ),
        ^^
    )
    ^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/checkpoint_manager.py", line 1474, in save
    self._checkpointer.save(
    ~~~~~~~~~~~~~~~~~~~~~~~^
        save_directory, args=args, custom_metadata=custom_metadata, force=True
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/checkpointer.py", line 259, in save
    self._handler.save(tmpdir.get(), args=ckpt_args)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/composite_checkpoint_handler.py", line 737, in save
    asyncio_utils.run_sync(async_save())
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/asyncio_utils.py", line 36, in run_sync
    return asyncio.run(coro)
           ~~~~~~~~~~~^^^^^^
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/composite_checkpoint_handler.py", line 735, in async_save
    f.result()
    ~~~~~~~~^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/futures/future.py", line 440, in result
    return self._f.result(timeout=timeout)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/futures/future.py", line 389, in result
    return self._t.result(timeout=timeout)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/futures/future.py", line 338, in result
    self.join(timeout=timeout)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/futures/future.py", line 335, in join
    raise self._exception
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/futures/future.py", line 311, in run
    super().run()
    ~~~~~~~~~~~^^
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/threading.py", line 994, in run
    self._target(*self._args, **self._kwargs)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/futures/future.py", line 256, in _target_setting_result
    self._result = target()
                   ~~~~~~^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/futures/future.py", line 382, in <lambda>
    target=lambda: asyncio_utils.run_sync(coro),
                   ~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/asyncio_utils.py", line 36, in run_sync
    return asyncio.run(coro)
           ~~~~~~~~~~~^^^^^^
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/json_checkpoint_handler.py", line 60, in _save_fn
    path.write_text(json.dumps(x))
                    ~~~~~~~~~~^^^
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/json/__init__.py", line 231, in dumps
    return _default_encoder.encode(obj)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/json/encoder.py", line 200, in encode
    chunks = self.iterencode(o, _one_shot=True)
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/json/encoder.py", line 261, in iterencode
    return _iterencode(o, 0)
  File "/home/gg/.local/share/uv/python/cpython-3.13.7-linux-x86_64-gnu/lib/python3.13/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
                    f'is not JSON serializable')
TypeError: Object of type ArrayImpl is not JSON serializable
INFO:absl:[process=0][thread=array_type_handler] Wrote 170 array_metadata.ArrayMetadata to /home/gg/git_repos/notebooks/vision_transformer/checkpoints/0.orbax-checkpoint-tmp/state.orbax-checkpoint-tmp/array_metadatas/process_0