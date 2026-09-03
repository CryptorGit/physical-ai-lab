# Aborted run — do not use

This pilot was stopped during startup validation. The frozen shallow source
initialized `backward_residual_scale` to `0.0`, so policy actions could not
change reverse motor targets. No completed params or adoption candidate exists.

The successor trainer requires an explicit positive residual scale and rejects
values outside `(0, 0.25]`.
