Traceback (most recent call last):
  File "/home/gg/git_repos/notebooks/vision_transformer/main.py", line 97, in <module>
    main()
    ~~~~^^
  File "/home/gg/git_repos/notebooks/vision_transformer/main.py", line 50, in main
    registry.add("state", ocp.PyTreeCheckpointHandler())
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/handler_registration.py", line 193, in add
    handler_to_register = checkpoint_args.get_registered_handler_cls(args)
  File "/home/gg/git_repos/notebooks/vision_transformer/.venv/lib/python3.13/site-packages/orbax/checkpoint/checkpoint_args.py", line 129, in get_registered_handler_cls
    raise TypeError(f'{arg} must be a subclass of `CheckpointArgs`.')
TypeError: <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler'> must be a subclass of `CheckpointArgs`.