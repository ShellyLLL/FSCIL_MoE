"""FSCIL session metrics: include base session, retain unrounded accuracies."""
import math


def session_metrics(accuracies):
    values = [float(x) for x in accuracies]
    if not values or any(not math.isfinite(x) or not 0 <= x <= 100 for x in values):
        raise ValueError("Expected a nonempty sequence of finite percentage accuracies.")
    return {"AA": math.fsum(values) / len(values), "PD": values[0] - values[-1],
            "sessions": len(values)}
