1. mplang/core/expr/ast.py:135: in _compute_mptypes
    raise ValueError(
E   ValueError: Specified rmask 4 is not a subset of deduced pmask 0.

A common cause of the above error is that on device A you try to compute using data from device B.

To fix this, you can either:

- Do the operation on device B
- Or send the data from device B to device A

example solution:
```python
y_pred_no_intercept = simp.revealTo(agg.sum(y_pred_party), model.label_party)

if model.intercept is not None:
    return simp.runAt(model.label_party, lambda x, b: x + b)(
        y_pred_no_intercept, model.intercept
    )
```