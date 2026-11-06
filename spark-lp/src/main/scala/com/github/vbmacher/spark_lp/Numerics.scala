package com.github.vbmacher.spark_lp

import com.github.vbmacher.spark_lp.dsl.LpModelException

/** Numeric validation shared by model construction, compilation, and solving. */
private[spark_lp] object Numerics {

  /** True when `value` is neither NaN nor infinite. */
  def isFinite(value: Double): Boolean = !value.isNaN && !value.isInfinite

  /** Homogeneous residual; zero equations have zero error, nonfinite scales never certify. */
  def relativeResidual(violation: Double, scale: Double): Double =
    if (!isFinite(violation) || !isFinite(scale)) Double.PositiveInfinity
    else if (scale > 0.0) violation / scale
    else if (violation == 0.0) 0.0
    else Double.PositiveInfinity

  /** Returns `value`, or throws [[LpModelException]] when it is NaN or infinite. */
  def requireFinite(value: Double, description: String): Unit =
    if (!isFinite(value)) throw new LpModelException(s"$description must be finite")
}
