"""Non-trainable Stage 0 placeholder; intentionally disconnected."""

NOT_IMPLEMENTED_STAGE_0 = "NOT_IMPLEMENTED_STAGE_0"


class TransitionBridge:
    """Interface marker with no parameters and no production action path."""

    trainable = False
    production_connected = False

    def __call__(self, *_args, **_kwargs) -> str:
        return NOT_IMPLEMENTED_STAGE_0

    def forward(self, *_args, **_kwargs) -> str:
        return NOT_IMPLEMENTED_STAGE_0
