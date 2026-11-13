# Numeric: banded Jacobian

The balance solver is a Newton-type method. At each iteration it needs the Jacobian of the objective vector with respect to the state vector. This matrix is built by finite differences, which normally costs one model evaluation per variable.

When the model is **banded**, most of those evaluations can be shared.

## Band assumption

A model has half-bandwidth `h` if objective `i` depends on state variable `j` only when `|i - j| <= h`. The Jacobian is then zero outside the band, and its nonzero entries sit within `h` of the diagonal.

The `h` values are set in the configuration (`options.solver.acceleration.banded_jacobian`): `node_band_size` (default 3) for the node model and `load_band_size` (default 1) for the load model. These defaults are also the minimum accepted values: setting a lower value raises a `ValueError`.

## Column grouping

Two columns `j` and `k` share a nonzero row only if `|j - k| <= 2h`. Columns spaced `2h + 1` apart therefore never overlap, so they can be perturbed together in one model evaluation.

The solver uses `2h + 1` groups. Group `g` perturbs all columns `j` with `j mod (2h + 1) = g`. Each row receives at most one perturbed column from that group, so its finite difference is attributable to that column alone. For column `j`, only rows `j - h` to `j + h` are copied from the group's result.

The number of model evaluations is `2h + 1`, independent of the number of variables `n`. When `2h + 1 >= n`, the dense Jacobian is computed instead, since banding saves nothing.

## Example

Take `n = 8` and `h = 1`, so there are 3 groups. Each cell shows the group of the column with a nonzero entry; `.` is structurally zero.

```
              col: 0  1  2  3  4  5  6  7
row 0                0  1  .  .  .  .  .  .
row 1                0  1  2  .  .  .  .  .
row 2                .  1  2  0  .  .  .  .
row 3                .  .  2  0  1  .  .  .
row 4                .  .  .  0  1  2  .  .
row 5                .  .  .  .  1  2  0  .
row 6                .  .  .  .  .  2  0  1
row 7                .  .  .  .  .  .  0  1

Evaluations:
  group 0 -> perturb columns {0, 3, 6}
  group 1 -> perturb columns {1, 4, 7}
  group 2 -> perturb columns {2, 5}
```

Here 3 evaluations replace 8. In each row the labels are distinct, which is what makes the recovery unambiguous.

For the node model (`h = 3`), the step is 7, so the Jacobian needs 7 evaluations regardless of the number of nodes.

## Notes

- The banded path assumes the band structure holds. If the model's dependencies are wider than declared, the Jacobian will be silently wrong.
- Setting `options.solver.acceleration.banded_jacobian.active` to `False` forces the dense computation.
