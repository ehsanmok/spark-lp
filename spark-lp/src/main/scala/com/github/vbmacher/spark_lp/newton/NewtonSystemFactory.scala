package com.github.vbmacher.spark_lp.newton

import com.github.vbmacher.spark_lp.vectors.DMatrix

/**
  * Builds the normal-equations system `B^T diag(w) B + Rd` for one interior-point iteration. With no
  * weights, uses `(I + Rp)^(-1)` for initialization. Implementations range from the direct
  * [[CholeskyFactory]] to the matrix-free [[CgFactory]]; the solver reuses a single factory across
  * iterations, so the accessors below report cumulative inner-solver work.
  */
private[spark_lp] trait NewtonSystemFactory {

  /**
    * Prepares a reusable [[NewtonSystem]] for the weighted Gramian of the current iteration.
    *
    * @param B       the constraint matrix `B`, its rows partitioned like the solver's vectors.
    * @param m       the number of equality-form constraint rows, i.e. the system dimension.
    * @param weights the iteration's scaling weights, or `None` for the unweighted initialization system.
    * @return a prepared system that solves any right-hand side of dimension `m`.
    */
  def build(B: DMatrix, m: Int, weights: Option[Weights]): NewtonSystem

  /** Maximum primal regularization `Rp` folded into the operator; `0.0` when unregularized. */
  def primalRegularization: Double = 0.0

  /** Maximum dual regularization `Rd` added to the Gramian diagonal; `0.0` when unregularized. */
  def dualRegularization: Double = 0.0

  /** Total inner-solver iterations across every system built (e.g. CG steps); `0` for direct solvers. */
  def innerIterations: Int = 0

  /** Total inner-solver restarts across every system built (e.g. CG cycle restarts); `0` for direct solvers. */
  def innerRestarts: Int = 0

  /** Peak preconditioner rank reached across every system built; `0` for solvers without a rank-based preconditioner. */
  def maximumRank: Int = 0

  /** Cooperative cancellation hook; throws to abort the solve when the injected monitor requests a stop. */
  def check(): Unit = ()
}
