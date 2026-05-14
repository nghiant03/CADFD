"""Concrete fault injector implementations.

Each class implements a specific fault injection strategy.
New fault types can be added by subclassing BaseFaultInjector
and registering with the fault registry.

Per-event randomization is supported for fault parameters via
range tuples (e.g. ``magnitude_range``, ``drift_rate_range``).
Per-mote relative scaling is supported by reading optional
``_mote_std`` / ``_mote_median`` keys that the orchestrator
:class:`CESTA.injection.injector.FaultInjector` injects into
``params`` before calling :meth:`apply`. Sigma-relative ranges
(e.g. ``magnitude_sigma_range``) override absolute ranges when
present and are interpreted as multipliers on ``_mote_std``.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from CESTA.injection.base import BaseFaultInjector


def _sample_range(
    params: dict[str, Any],
    range_key: str,
    scalar_key: str | None,
    default: tuple[float, float],
    rng: np.random.Generator,
) -> float:
    """Sample a scalar from a range param, with scalar/back-compat fallback."""
    rng_val = params.get(range_key)
    if rng_val is not None:
        lo, hi = float(rng_val[0]), float(rng_val[1])
        return float(rng.uniform(lo, hi))
    if scalar_key is not None:
        scalar_val = params.get(scalar_key)
        if scalar_val is not None:
            return float(scalar_val)
    lo, hi = default
    return float(rng.uniform(lo, hi))


class SpikeFaultInjector(BaseFaultInjector):
    """Injects spike faults: sudden large deviations from normal values.

    Parameters:
        magnitude_range: ``(min, max)`` absolute offset sampled per event.
        magnitude_sigma_range: ``(k_min, k_max)`` multipliers on the mote
            baseline std (overrides ``magnitude_range`` when present and
            ``_mote_std`` is available).
    """

    @property
    def fault_name(self) -> str:
        return "SPIKE"

    def apply(
        self,
        data: NDArray[np.float64],
        mask: NDArray[np.bool_],
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return data

        sigma_range = params.get("magnitude_sigma_range")
        mote_std = params.get("_mote_std")
        segments = self._find_contiguous_segments(indices)

        for segment in segments:
            if sigma_range is not None and mote_std is not None and mote_std > 0:
                k = float(rng.uniform(float(sigma_range[0]), float(sigma_range[1])))
                magnitude = abs(k * float(mote_std))
            else:
                magnitude = abs(
                    _sample_range(
                        params,
                        "magnitude_range",
                        scalar_key=None,
                        default=(2.5, 5.0),
                        rng=rng,
                    )
                )
            sign = float(rng.choice([-1.0, 1.0]))
            spike_value = magnitude * sign
            for idx in segment:
                data[idx] += spike_value

        return data


class DriftFaultInjector(BaseFaultInjector):
    """Injects drift faults: gradual linear trend over fault duration.

    Parameters:
        drift_rate_range: ``(min, max)`` absolute rate per timestep, sampled
            per event.
        drift_rate_sigma_range: ``(k_min, k_max)`` multipliers on the mote
            baseline std (per timestep), overrides absolute when ``_mote_std``
            is available.
        drift_rate: legacy scalar, used if range params are absent.
    """

    @property
    def fault_name(self) -> str:
        return "DRIFT"

    def apply(
        self,
        data: NDArray[np.float64],
        mask: NDArray[np.bool_],
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return data

        sigma_range = params.get("drift_rate_sigma_range")
        mote_std = params.get("_mote_std")
        segments = self._find_contiguous_segments(indices)

        for segment in segments:
            if sigma_range is not None and mote_std is not None and mote_std > 0:
                k = float(rng.uniform(float(sigma_range[0]), float(sigma_range[1])))
                drift_rate = k * float(mote_std)
            else:
                drift_rate = _sample_range(
                    params,
                    "drift_rate_range",
                    scalar_key="drift_rate",
                    default=(0.05, 0.15),
                    rng=rng,
                )
            direction = float(rng.choice([-1.0, 1.0]))
            for i, idx in enumerate(segment):
                data[idx] += direction * drift_rate * (i + 1)

        return data


class StuckFaultInjector(BaseFaultInjector):
    """Injects stuck faults: value freezes at the start of the fault.

    Parameters:
        jitter_sigma_factor: optional float; when set together with
            ``_mote_std``, adds Gaussian noise with std
            ``jitter_sigma_factor * _mote_std`` around the frozen value to
            simulate subtle stuck-with-noise behavior.
    """

    @property
    def fault_name(self) -> str:
        return "STUCK"

    def apply(
        self,
        data: NDArray[np.float64],
        mask: NDArray[np.bool_],
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return data

        jitter_factor = params.get("jitter_sigma_factor")
        mote_std = params.get("_mote_std")
        jitter_std = (
            float(jitter_factor) * float(mote_std)
            if jitter_factor is not None and mote_std is not None and mote_std > 0
            else 0.0
        )

        segments = self._find_contiguous_segments(indices)

        for segment in segments:
            stuck_value = data[segment[0]]
            for idx in segment:
                if jitter_std > 0.0:
                    data[idx] = stuck_value + float(rng.normal(0.0, jitter_std))
                else:
                    data[idx] = stuck_value

        return data
