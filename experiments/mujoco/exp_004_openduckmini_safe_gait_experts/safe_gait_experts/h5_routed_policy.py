"""H5 command-domain routing with target-space composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .h4_post_training import (
    H4_ACTOR_OBSERVATION_WIDTH,
    infer_h4_action_numpy,
    mask_h4_head_action,
)
from .h5_target_contract import (
    H5_ACTION_WIDTH,
    H5_DOMAINS,
    h5_blend_targets,
    h5_decode_absolute_targets,
    h5_domain_for_route,
    h5_final_guard_step,
)


@dataclass(frozen=True)
class H5DomainCandidate:
    """A command-conditioned actor candidate for one H5 domain."""

    domain: str
    params: Any
    params_sha256: str | None = None
    manifest_sha256: str | None = None
    training_manifest_sha256: str | None = None
    training_resolved_config_sha256: str | None = None
    training_command_contract_id: str | None = None
    training_command_mapper: str | None = None
    training_command_mapper_inferred_from_v2_contract: bool = False

    def validate(self) -> None:
        if self.domain not in H5_DOMAINS:
            raise ValueError(f"unsupported H5 candidate domain: {self.domain!r}")
        if self.params is None:
            raise ValueError(f"H5 {self.domain} candidate parameters are missing")
        for label, value in (
            ("params_sha256", self.params_sha256),
            ("manifest_sha256", self.manifest_sha256),
            ("training_manifest_sha256", self.training_manifest_sha256),
            (
                "training_resolved_config_sha256",
                self.training_resolved_config_sha256,
            ),
        ):
            if value is not None and (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"H5 {label} must be a lowercase SHA256")
        for label, value in (
            ("training_command_contract_id", self.training_command_contract_id),
            ("training_command_mapper", self.training_command_mapper),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"H5 {label} must be a non-empty string when present")


@dataclass(frozen=True)
class H5RouteStep:
    """Auditable result of one route composition and final guard step."""

    from_role: str
    to_role: str
    from_domain: str
    to_domain: str
    alpha: float
    from_action: tuple[float, ...]
    to_action: tuple[float, ...]
    from_targets: tuple[float, ...]
    to_targets: tuple[float, ...]
    blended_targets: tuple[float, ...]
    applied_targets: tuple[float, ...]
    target_space_blend: bool = True
    final_guard_call_count: int = 1


class H5RoutedPolicyBank:
    """Route planar/reverse actors and compose them in target space.

    The base bank remains available for the explicit legacy fallback during
    diagnostics.  Once both H5 candidates are installed, every route in the
    H5 partition is inferred by a domain candidate and no raw-action blend is
    used at a switch.
    """

    def __init__(
        self,
        base_bank: Any,
        candidates: Mapping[str, H5DomainCandidate],
        *,
        allow_legacy_fallback: bool = False,
    ) -> None:
        self.base_bank = base_bank
        self.allow_legacy_fallback = bool(allow_legacy_fallback)
        self.candidates = {
            str(domain): candidate for domain, candidate in candidates.items()
        }
        if set(self.candidates) != set(H5_DOMAINS):
            raise ValueError("H5 routed bank requires exactly planar and reverse candidates")
        for candidate in self.candidates.values():
            candidate.validate()
        self.inference_counts: dict[str, int] = {domain: 0 for domain in H5_DOMAINS}
        self.legacy_fallback_count = 0
        self.legacy_fallback_roles: dict[str, int] = {}
        self.last_step: H5RouteStep | None = None
        # The frozen routed schedule needs the raw action for action-history
        # observations while its simulator hook consumes ``last_step``'s
        # decoded target.  Release code must leave this false.
        self.integration_return_raw_action = False

    def _infer_base(self, role: str, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (H4_ACTOR_OBSERVATION_WIDTH,):
            raise ValueError("H5 actor observations must be exactly 116-wide")
        if not hasattr(self.base_bank, "infer"):
            raise ValueError("H5 legacy fallback bank has no infer method")
        try:
            action = self.base_bank.infer(role, values)
        except (TypeError, ValueError):
            action = self.base_bank.infer(role, values[:101])
        result = np.asarray(action, dtype=np.float64)
        if result.shape != (H5_ACTION_WIDTH,):
            raise ValueError("legacy fallback action must be exactly 14-wide")
        return result

    def infer(self, role: str, observation: np.ndarray) -> np.ndarray:
        route = str(role)
        domain = h5_domain_for_route(route)
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (H4_ACTOR_OBSERVATION_WIDTH,):
            raise ValueError("H5 actor observations must be exactly 116-wide")
        action = infer_h4_action_numpy(self.candidates[domain].params, values)
        action = mask_h4_head_action(action)
        self.inference_counts[domain] += 1
        return np.asarray(action, dtype=np.float64)

    def infer_or_legacy(self, role: str, observation: np.ndarray) -> np.ndarray:
        """Infer an H5 route, optionally retaining a diagnostic legacy fallback.

        Strict H5 evaluation must fail at the actor boundary.  Falling back to
        a legacy ONNX policy after an H5 ``ValueError`` can make a broken
        candidate look healthy and makes provenance impossible to interpret.
        The compatibility path remains opt-in for old exploratory callers.
        """

        try:
            return self.infer(role, observation)
        except ValueError:
            if not self.allow_legacy_fallback:
                raise
            self.legacy_fallback_count += 1
            self.legacy_fallback_roles[str(role)] = (
                self.legacy_fallback_roles.get(str(role), 0) + 1
            )
            return self._infer_base(role, observation)

    def infer_route(
        self,
        decision: Any,
        observation: np.ndarray,
        previous_applied_targets: np.ndarray | None = None,
        *,
        apply_guard: bool = True,
        return_raw_action: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Infer both sides, decode both, blend targets, then optionally guard.

        The frozen schedule expects ``(raw_action, applied_action)``.  H5 keeps
        those action-history values explicit and publishes the decoded desired
        target through ``last_step`` for the simulator hook; the schedule then
        owns the one final target guard.
        """

        from_role = str(decision.blend_from_expert)
        to_role = str(decision.blend_to_expert)
        from_domain = h5_domain_for_route(from_role)
        to_domain = h5_domain_for_route(to_role)
        from_action = self.infer_or_legacy(from_role, observation)
        to_action = (
            from_action
            if to_role == from_role
            else self.infer_or_legacy(to_role, observation)
        )
        from_targets = h5_decode_absolute_targets(from_action, domain=from_domain)
        to_targets = h5_decode_absolute_targets(to_action, domain=to_domain)
        blended = h5_blend_targets(
            from_targets, to_targets, float(decision.blend_alpha)
        )
        previous = (
            np.zeros(H5_ACTION_WIDTH, dtype=np.float64)
            if previous_applied_targets is None
            else previous_applied_targets
        )
        applied = h5_final_guard_step(blended, previous)
        step = H5RouteStep(
            from_role=from_role,
            to_role=to_role,
            from_domain=from_domain,
            to_domain=to_domain,
            alpha=float(decision.blend_alpha),
            from_action=tuple(float(value) for value in from_action),
            to_action=tuple(float(value) for value in to_action),
            from_targets=tuple(float(value) for value in from_targets),
            to_targets=tuple(float(value) for value in to_targets),
            blended_targets=tuple(float(value) for value in blended),
            applied_targets=tuple(float(value) for value in applied),
        )
        self.last_step = step
        raw_action = (1.0 - step.alpha) * np.asarray(step.from_action) + step.alpha * np.asarray(step.to_action)
        raw_action = np.asarray(raw_action, dtype=np.float64)
        raw_action[list(range(5, 9))] = 0.0
        del apply_guard, return_raw_action, applied
        return raw_action, raw_action.copy()

    def manifest(self) -> dict[str, Any]:
        return {
            "contract_id": "OPEN_DUCK_MINI_H5_TARGET_SPACE_ROUTING_V1",
            "candidate_domains": sorted(self.candidates),
            "candidate_provenance": {
                domain: {
                    "params_sha256": candidate.params_sha256,
                    "manifest_sha256": candidate.manifest_sha256,
                    "training_manifest_sha256": candidate.training_manifest_sha256,
                    "training_resolved_config_sha256": (
                        candidate.training_resolved_config_sha256
                    ),
                    "training_command_contract_id": (
                        candidate.training_command_contract_id
                    ),
                    "training_command_mapper": candidate.training_command_mapper,
                    "training_command_mapper_inferred_from_v2_contract": (
                        candidate.training_command_mapper_inferred_from_v2_contract
                    ),
                }
                for domain, candidate in sorted(self.candidates.items())
            },
            "composition_order": "decode_each_candidate_then_blend_targets_then_one_guard",
            "inference_counts": dict(self.inference_counts),
            "legacy_fallback": {
                "allowed": self.allow_legacy_fallback,
                "count": self.legacy_fallback_count,
                "roles": dict(sorted(self.legacy_fallback_roles.items())),
                "strict_fail_closed": not self.allow_legacy_fallback,
            },
            "hardware_deployment": "PROHIBITED",
        }
