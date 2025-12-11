import orbax.checkpoint as ocp
from flax.training import train_state

from utils.config import Config


class CheckpointManager:
    def __init__(self, config: Config):
        self.config = config
        self.checkpoint_options = ocp.CheckpointManagerOptions(
            max_to_keep=config.MAX_TO_KEEP,
            save_interval_steps=config.SAVE_INTERVAL,
            enable_async_checkpointing=config.ASYNC_CHECKPOINTING,
            best_fn=config.BEST_FN,
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
            directory=self.config.CHECKPOINT_PATH.resolve(),
            handler_registry=self.registry,
            options=self.checkpoint_options,
        )

    def get_manager(self):
        return self.manager

    def restore(self, state, logging) -> tuple[train_state.TrainState, int]:
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
