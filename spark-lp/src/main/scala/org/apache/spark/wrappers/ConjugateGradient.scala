package org.apache.spark.wrappers

/** Matrix-free preconditioned CG for one symmetric positive-definite system.
  * Vector kernels use netlib BLAS; no dense matrix is formed. Inputs are copied.
  * The operator and preconditioner must leave their input unchanged and return a
  * vector of the same length. Calls are synchronous; they must not retain inputs.
  * Optional component tolerances replace the norm threshold for convergence; reported norms
  * remain Euclidean. This mutable solver is driver-local and is not thread-safe.
  */
final class ConjugateGradient(
  rhs: Array[Double],
  operator: Array[Double] => Array[Double],
  relativeTolerance: Double,
  absoluteTolerance: Double = 0.0,
  initialGuess: Option[Array[Double]] = None,
  componentTolerances: Option[Array[Double]] = None) {

  require(relativeTolerance > 0.0 && !relativeTolerance.isInfinite,
    "CG tolerance must be finite and positive")
  require(absoluteTolerance >= 0.0 && !absoluteTolerance.isInfinite,
    "CG absolute tolerance must be finite and nonnegative")
  require(rhs.forall(v => !v.isNaN && !v.isInfinite), "CG right-hand side must be finite")
  require(initialGuess.forall(x => x.length == rhs.length && x.forall(v => !v.isNaN && !v.isInfinite)),
    "CG initial guess must be finite and match the right-hand side")
  require(componentTolerances.forall(t => t.length == rhs.length && t.forall(v => v > 0.0 && !v.isInfinite)),
    "CG component tolerances must be finite, positive and match the right-hand side")

  private val blas = NativeNetlib.blas
  private val n = rhs.length
  private val b = rhs.clone()
  private val residualTolerances = componentTolerances.map(_.clone())
  val rhsNorm: Double = blas.dnrm2(n, b, 1)
  val targetNorm: Double = math.max(relativeTolerance * rhsNorm, absoluteTolerance)
  private val x = if (rhsNorm == 0.0) new Array[Double](n)
    else initialGuess.map(_.clone()).getOrElse(new Array[Double](n))
  private var r = if (rhsNorm != 0.0 && initialGuess.nonEmpty) trueResidual() else b.clone()
  private var residual = blas.dnrm2(n, r, 1)
  private var totalSteps = 0
  private var cycles = 0

  def solution: Array[Double] = x.clone()
  def residualNorm: Double = residual
  def converged: Boolean = residualTolerances match {
    case Some(targets) => r.indices.forall(i => math.abs(r(i)) <= targets(i))
    case None => residual <= targetNorm
  }
  def iterations: Int = totalSteps
  def restarts: Int = math.max(0, cycles - 1)

  /** Runs with a fixed preconditioner and step budget, retaining state for a retry
    * with a stronger preconditioner. Acceptance always uses the true residual.
    * `reportResidual` receives (isTrueResidual, norm); it may throw to stop the solve.
    * Optional true-residual probes do not restart conjugate directions.
    * A return without convergence leaves acceptance or retry policy to the caller.
    */
  def solve(
    precondition: Array[Double] => Array[Double],
    maxIterations: Int,
    reportResidual: (Boolean, Double) => Unit = (_, _) => (),
    checkTrueResidual: Int => Boolean = _ => false): Unit = {
    require(maxIterations > 0, "CG iteration budget must be positive")
    var stepsAtRank = 0
    var gaveUp = false
    var finished = converged

    while (!finished && !gaveUp && stepsAtRank < maxIterations) {
      cycles += 1
      val cycleStartNorm = residual
      var z = precondition(r)
      val p = z.clone()
      var rz = blas.ddot(n, r, 1, z, 1)
      var cycleBest = residual
      var stepsSinceImprovement = 0
      var stagnated = false

      while (!finished && !stagnated && stepsAtRank < maxIterations) {
        val ap = operator(p)
        val pAp = blas.ddot(n, p, 1, ap, 1)
        if (pAp.isNaN || pAp.isInfinite)
          throw new IllegalStateException(s"Non-finite curvature in the CG solve at step ${totalSteps + 1}")
        if (pAp <= 0.0)
          throw new IllegalStateException(
            s"CG operator is not positive definite (p^T A p = $pAp at CG step ${totalSteps + 1})")
        val alpha = rz / pAp
        blas.daxpy(n, alpha, p, 1, x, 1)
        blas.daxpy(n, -alpha, ap, 1, r, 1)
        residual = blas.dnrm2(n, r, 1)
        totalSteps += 1
        if (residual.isNaN || residual.isInfinite)
          throw new IllegalStateException("Non-finite CG residual")
        stepsAtRank += 1
        finished = converged
        reportResidual(false, residual)

        if (checkTrueResidual(totalSteps) && !finished) {
          val actual = trueResidual()
          val actualNorm = blas.dnrm2(n, actual, 1)
          finished = residualTolerances match {
            case Some(targets) => actual.indices.forall(i => math.abs(actual(i)) <= targets(i))
            case None => actualNorm <= targetNorm
          }
          if (finished) { r = actual; residual = actualNorm }
          reportResidual(true, actualNorm)
        }
        if (!finished) {
          if (residual < 0.95 * cycleBest) {
            cycleBest = residual
            stepsSinceImprovement = 0
          } else {
            stepsSinceImprovement += 1
            stagnated = stepsSinceImprovement >= 25
          }
          if (!stagnated) {
            z = precondition(r)
            val rzNew = blas.ddot(n, r, 1, z, 1)
            blas.dscal(n, rzNew / rz, p, 1)
            blas.daxpy(n, 1.0, z, 1, p, 1)
            rz = rzNew
          }
        }
      }

      // Replace the recurrence residual before accepting the solution or restarting.
      r = trueResidual()
      residual = blas.dnrm2(n, r, 1)
      finished = converged
      reportResidual(true, residual)
      gaveUp = !finished && residual > 0.9 * cycleStartNorm
    }
  }

  private def trueResidual(): Array[Double] = {
    val result = b.clone()
    blas.daxpy(n, -1.0, operator(x), 1, result, 1)
    result
  }
}
