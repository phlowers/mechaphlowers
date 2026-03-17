# Residual Rated Tensile Strength (RRTS)

## Context

When strands are cut — due to damage, corrosion, or mechanical incidents — the rated tensile strength (RTS) of a cable is reduced. RRTS quantifies the remaining capacity after accounting for those cuts.

Strand cuts are declared **per layer**. Each layer has a unit RTS per strand (`rts_layer_i`), so the total strength loss is the sum of individual strand contributions across all affected layers.

## Scientific formulas

The RRTS of a damaged cable:

$$
RRTS = RTS_{cable} - \sum_{i=1}^{N_{layers}} n_{cut,i} \cdot rts_{layer,i}
$$

The utilization rate $\tau$ (as a percentage) relative to the safety-adjusted RRTS:

$$
\tau\,(\%) = \frac{T_{max}}{RRTS \cdot k_s} \times 100
$$

| Symbol | Description |
|--------|-------------|
| $RTS_{cable}$ | Rated Tensile Strength of the intact cable (N) |
| $n_{cut,i}$ | Number of cut strands in layer $i$ |
| $rts_{layer,i}$ | Unit RTS per strand in layer $i$ (N); `0` for unused layers |
| $N_{layers}$ | Number of layers (up to 8) |
| $T_{max}$ | Maximum mechanical tension applied (N) |
| $k_s$ | Safety coefficient (dimensionless, catalog column `safety_coefficient`) |

When the [`high_safety`][mechaphlowers.entities.arrays.CableArray.high_safety] flag is enabled, an additional security factor of 1.5 is applied to $k_s$:

$$
k_{s,\text{eff}} = k_s \times 1.5
$$

The effective value is exposed by the read-only [`safety_coefficient`][mechaphlowers.entities.arrays.CableArray.safety_coefficient] property and is used automatically by [`utilization_rate`][mechaphlowers.entities.arrays.CableArray.utilization_rate].

!!! Warning "Model assumptions"

    - The RRTS model assumes that all strands in a given layer have the same RTS. Heterogeneous layers must be pre-processed (e.g., split or averaged) before loading data.
    
    - The utilization rate calculation assumes that the maximum tension $T_{max}$ is applied uniformly across the cable section, which may not reflect localized stress concentrations in real-world scenarios.

    - The first layer is the external layer and the last layer is also called central layer.

## Catalog parameters

The following columns must be present in the cable catalog CSV for RRTS support:

| Column | Unit | Description |
|--------|------|-------------|
| `rts_cable` | N | RTS of the intact cable |
| `rts_layer_1` … `rts_layer_8` | N | Unit RTS per strand per layer (`0` = unused) |
| `safety_coefficient` | — | Safety coefficient $k_s$ |
| `nb_strand_layer_1` … `nb_strand_layer_8` | — | Number of strands per layer (`0` = unused) |

!!! note "Coverage check"
    The ratio $\frac{RTS_{cable}}{\sum_{i} rts_{layer,i} \times nb_{strand,i}} \times 100$ indicates how well the
    strand-level model explains the cable RTS. An acceptable value is between 75% and 100%.

!!! warning "Layer-level assumption"
    The model assumes all strands in a given layer are identical (same RTS). Layers with heterogeneous strands must be split or averaged externally before loading data.

## Usage

```python
import numpy as np
from mechaphlowers.data.catalog import sample_cable_catalog

cable = sample_cable_catalog.get_as_object(["ASTER600"])

# Declare cut strands per layer — here 1 strand cut in layer 3
# Maximum allowed per layer: int(nb_strand_layer_i / 2)
cable.cut_strands = np.array([0, 0, 1, 0, 0, 0, 0, 0])

# RRTS for the whole section (scalar — applies uniformly to all spans)
print(cable.rrts)
# 196800.0

# Utilization rate (%) for each span, given the maximum tension per span
tensions = np.array([50000.0, 75000.0, 90000.0])  # N, one value per span
print(cable.utilization_rate(tensions))
# [16.93766938 25.40650407 30.48780488]

# Activate the high safety mode — safety_coefficient is multiplied by 1.5
cable.high_safety = True
print(cable.safety_coefficient)     # catalog value × 1.5
print(cable.utilization_rate(tensions))
# tighter limits — values are divided by 1.5 compared to the default
```

## Architecture

Under the hood, `CableArray` delegates all RRTS-related computations to a
[`ITensileStrength`][mechaphlowers.core.models.cable.cable_strength.ITensileStrength]
implementation. The default implementation is
[`AdditiveLayerRts`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts],
which computes strand-level losses additively per layer. A custom model can be
injected via the `tensile_strength` argument of `CableArray`.

!!! note "Custom tensile strength models"
    You can implement your own tensile strength model by subclassing
    [`ITensileStrength`][mechaphlowers.core.models.cable.cable_strength.ITensileStrength].
    This interface defines the contract for `cut_strands`, `high_safety`,
    `safety_coefficient`, `nb_strand_per_layer`, `rts_coverage`, `rrts`, and
    `utilization_rate`. Once implemented, pass an instance to `CableArray`:

    ```python
    cable = CableArray(data, tensile_strength=MyCustomModel(data))
    ```

    This is useful when the default additive per-layer model does not match
    your cable technology (e.g. composite cables, non-uniform strand
    contributions, or proprietary strength models).

See the [CableArray API reference](../docstring/entities/arrays.md) and the
[Cable Strength API reference](../docstring/core/models/cable/cable_strength.md)
for full documentation.
