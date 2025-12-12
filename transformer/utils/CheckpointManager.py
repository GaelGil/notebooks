import orbax.checkpoint as ocp
from flax import nnx

from pathlib import Path


class CheckpointManager:
    def __init__(
        self,
        max_to_keep: int,
        save_interval_steps: int,
        async_checkpointing: bool,
        best_fn: str,
        checkpoint_path: Path,
    ):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
            enable_async_checkpointing=async_checkpointing,
            best_fn=best_fn,
        )
        self.registry = ocp.handlers.DefaultCheckpointHandlerRegistry()

        # Define the checkpoint manager
        self.manager = None

    def add_to_register(self, val: str, save_fn, restore_fn) -> None:
        """
        Add the save and restore functions to the registry along with the value
        Essentially the same as the below
        self.registry.add("state", ocp.args.StandardSave)
        self.registry.add("state", ocp.args.StandardRestore)

        Args:
            val: str
            save_fn: function
            restore_fn: function

        Returns:
            None
        """

        self.registry.add(val, save_fn)
        self.registry.add(val, restore_fn)

    def create_manager(self):
        """
        Creates the checkpoint manager using the registry options and checkpoint
        options

        Args:
            None

        Returns:
            None
        """
        self.manager = ocp.CheckpointManager(
            directory=self.checkpoint_path.resolve(),
            handler_registry=self.registry,
            options=self.checkpoint_options,
        )

    def get_manager(self):
        return self.manager

    def restore(self, state: nnx.TrainState, logging) -> tuple[nnx.TrainState, int]:
        # restore previous checkpoint
        if self.manager.latest_step():  # check if there is a latest checkpoint
            logging.info("Restoring from latest checkpoint")
            # get the best step/checkpoint
            # this was deinfed in the checkpoint options
            best_step = self.manager.best_step()
            # restore from the best step
            restored = self.manager.restore(
                step=best_step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(state),
                    metrics=ocp.args.JsonRestore(),
                ),
            )

            # update state to the restored state
            state = restored.state
        else:
            logging.info("No checkpoint found, training from scratch")
        return (
            state,
            self.manager.latest_step() if self.manager.latest_step() else 0,
        )
