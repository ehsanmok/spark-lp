package com.github.vbmacher.spark_lp.newton

import breeze.linalg.{DenseVector => BDV}
import com.github.vbmacher.spark_lp.{ProgressWindow, SolveMonitor, SolvePhase, SolveStopped, StopReason, WorkProgress}
import com.github.vbmacher.spark_lp.vectors.{DMatrix, DVector}
import com.github.vbmacher.spark_lp.vectors.dmatrix.implicits._
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.wrappers.ConjugateGradient

/**
  * Matrix-free [[NewtonSystemFactory]] applying preconditioned CG to `B^T W B + Rd`. Adaptive rank
  * starts with Jacobi and adds updated-diagonal Cholesky pivots only when a solve misses its target.
  * Rank is bounded even when explicitly requested. True (unpreconditioned) residuals decide
  * acceptance; exhausted solves may return at most a 1e-3 relative defect. The outer loop alone
  * judges original-LP convergence and certificates.
  *
  * @param relTolerance  relative CG stopping tolerance for each residual component; must be finite and positive.
  * @param maxIterations CG step budget per preconditioner rank; `<= 0` selects an `m`-derived default.
  * @param config        regularization and preconditioner rank/memory controls.
  * @param monitor       reports solver phases and progress and supplies cooperative cancellation.
  */
private[spark_lp] final class CgFactory(
  relTolerance: Double,
  maxIterations: Int,
  config: CgConfig = CgConfig(),
  monitor: SolveMonitor = new SolveMonitor())(implicit spark: SparkSession) extends NewtonSystemFactory with LazyLogging {

  import CgFactory._

  require(relTolerance > 0.0 && !relTolerance.isInfinite, "CG tolerance must be finite and positive")
  // The escalated rank persists across systems: once one iteration's system forces an
  // escalation, later iterations (which are at least as ill-conditioned) start from it.
  private var currentRank: Int = -1
  private var steps: Int = 0
  private var restarts: Int = 0
  private var peakRank: Int = 0
  override def check(): Unit = monitor.check()
  override val primalRegularization: Double = config.primalRegularization
  override val dualRegularization: Double = config.dualRegularization
  override def innerIterations: Int = steps
  override def innerRestarts: Int = restarts
  override def maximumRank: Int = peakRank

  override def build(B: DMatrix, m: Int, weights: Option[Weights]): NewtonSystem =
    new CgSystem(B, m, weights)

  private final class CgSystem(B: DMatrix, m: Int, weights: Option[Weights]) extends NewtonSystem {
    // Initialization uses the same regularized least-squares operator with S/X = I.
    private val initialWeight = 1.0 / (1.0 + primalRegularization)
    // A uniform initialization weight needs no extra RDD or second matrix traversal.
    private val w = weights.map(_.squared)
    private val operatorScale = if (weights.isDefined) 1.0 else initialWeight
    private val matrix = new DMatrixOps(B)
    private val requestedRank = config.preconditionerRank
    private val budgetRank = autoMaxRank(m, config.preconditionerMemoryBytes)
    private val maxRank = if (requestedRank > 0) math.min(requestedRank, budgetRank) else budgetRank
    if (currentRank < 0) currentRank = if (requestedRank > 0) maxRank else 0
    currentRank = math.min(currentRank, maxRank)
    monitor.phase(SolvePhase.SystemSetup)
    private val gramianDiagonal = matrix.gramianDiagonal(w).values.map(_ * operatorScale)
    private val dual = gramianDiagonal.map(d => dualRegularization * (if (d > 0.0) math.min(1.0, d) else 1.0))
    private val diagonal = gramianDiagonal.indices.map(i => gramianDiagonal(i) + dual(i)).toArray
    // Keep pivots and the unfloored Schur diagonal for this weighted system. Escalation
    // fetches only new columns; the factors must be rebuilt when the weights change.
    private val partial = new PartialCholesky(diagonal, 0, j => {
      monitor.check()
      val column = matrix.gramianColumns(Array(j), w)
      if (operatorScale != 1.0) {
        var i = 0
        while (i < column.length) { column(i) *= operatorScale; i += 1 }
      }
      column(j) += dual(j)
      column
    }, dual.min, completed => monitor.report(SolvePhase.SystemSetup,
      work = Some(WorkProgress(completed, preconditionerRank = Some(completed)))))
    private def extendPreconditioner(rank: Int): Unit = {
      partial.extendTo(rank)
      monitor.check()
      peakRank = math.max(peakRank, partial.rank)
    }
    extendPreconditioner(currentRank)

    private val maxIter = if (maxIterations > 0) maxIterations else math.min(math.max(100L, 2L * m), 1000L).toInt

    private var pBroadcast: Broadcast[DenseVector] = _
    private var lastSolution: Array[Double] = _

    /** One application of the operator: `B^T (w * (B p))`, driver memory O(m). */
    private def applyOperator(p: Array[Double]): Array[Double] = {
      monitor.check()
      if (pBroadcast != null) pBroadcast.destroy()
      // CG updates its vectors in place; the broadcast must own an immutable snapshot.
      pBroadcast = spark.sparkContext.broadcast(new DenseVector(p.clone()))
      val result = matrix.gramianProduct(pBroadcast, w).values
      var i = 0
      while (i < result.length) {
        result(i) = operatorScale * result(i) + dual(i) * p(i)
        i += 1
      }
      result
    }

    override def solve(rhs: DenseVector, absTolerance: Double): DenseVector = {
      monitor.phase(SolvePhase.InnerSolve)
      require(rhs.size == m, "CG right-hand side must match the operator")
      val smallest = rhs.values.map(math.abs).filter(_ > 0.0).reduceOption((a, b) => math.min(a, b)).getOrElse(1.0)
      val targets = rhs.values.map(v => math.max(absTolerance, relTolerance * math.max(smallest, math.abs(v))))
      val warmStarted = new ConjugateGradient(rhs.values, applyOperator, relTolerance, absTolerance,
        Option(lastSolution), Some(targets))
      // Reuse a previous solution only when it improves on the new right-hand side's zero start.
      val cg = if (lastSolution != null && warmStarted.residualNorm >= warmStarted.rhsNorm)
        new ConjugateGradient(rhs.values, applyOperator, relTolerance, absTolerance, componentTolerances = Some(targets))
      else warmStarted
      if (cg.rhsNorm == 0.0) return new DenseVector(cg.solution)

      val progress = monitor.control.stagnation.map(c => new ProgressWindow(c, c.innerPatience))
      def reportResidual(isTrue: Boolean, residual: Double): Unit = {
        monitor.report(SolvePhase.InnerSolve, work = Some(WorkProgress(cg.iterations,
          residual = Some(residual), trueResidual = isTrue, preconditionerRank = Some(partial.rank))))
        if (isTrue && !cg.converged) {
          val stalled = progress.exists(_.observe(cg.iterations, Vector(residual / cg.rhsNorm)))
          if (stalled || monitor.control.stagnation.exists(cg.iterations >= _.maxInnerSteps))
            throw SolveStopped(StopReason.NoProgress)
        }
      }
      def checkTrueResidual(totalSteps: Int): Boolean = monitor.control.stagnation.exists(c =>
        totalSteps % math.min(25, c.innerPatience) == 0 || totalSteps >= c.maxInnerSteps)

      try {
        reportResidual(isTrue = true, residual = cg.residualNorm)
        var exhausted = false
        while (!cg.converged && !exhausted) {
          cg.solve(r => partial(new BDV(r)).data, maxIter, reportResidual, checkTrueResidual)
          if (!cg.converged) {
            if (currentRank < maxRank) {
              currentRank = math.min(maxRank, math.max(DefaultPreconditionerRank, 2 * currentRank))
              logger.info(s"CG at residual ${cg.residualNorm} (target ${cg.targetNorm}) after ${cg.iterations} " +
                s"steps; escalating the partial Cholesky preconditioner to rank $currentRank")
              monitor.phase(SolvePhase.SystemSetup)
              extendPreconditioner(currentRank)
              monitor.phase(SolvePhase.InnerSolve)
            } else exhausted = true
          }
        }

        // A bounded inexact direction can still expose a Farkas certificate on the
        // next LP iterate. This is never an original-LP convergence test.
        if (cg.residualNorm.isNaN || cg.residualNorm.isInfinite ||
            (!cg.converged && cg.residualNorm > math.max(cg.targetNorm, 1e-3 * cg.rhsNorm))) {
          throw new IllegalStateException(
            s"Normal-equations CG solve stalled: residual ${cg.residualNorm} after ${cg.iterations} steps " +
              s"(target ${cg.targetNorm}, right-hand side norm ${cg.rhsNorm}). Check scaling, regularization, " +
              "cgTolerance/cgMaxIterations and preconditioner settings.")
        }
        logger.debug(s"CG residual ${cg.residualNorm} (target ${cg.targetNorm}) in ${cg.iterations} " +
          s"steps; inexact=${!cg.converged}")
        lastSolution = cg.solution
        new DenseVector(lastSolution.clone())
      } finally {
        steps += cg.iterations
        restarts += cg.restarts
      }
    }

    override def release(): Unit = {
      if (pBroadcast != null) pBroadcast.destroy()
    }
  }
}

private[spark_lp] object CgFactory {
  /**
    * Applies the regularized normal-equations operator `scale * (B^T diag(w) B) p + dual * p`
    * without materializing the Gramian, reusing the distributed matrix-vector products.
    *
    * @param matrix  the stored constraint matrix `B` wrapped for Gramian products.
    * @param weights optional squared iteration weights `w`; `None` applies the unweighted product.
    * @param p       broadcast operand vector.
    * @param dual    dual regularization added on the diagonal.
    * @param scale   scalar multiplying the Gramian product (folds in the initialization weight).
    * @return the operator applied to `p`, an `O(m)` driver vector.
    */
  private[spark_lp] def regularizedProduct(matrix: DMatrixOps,
    weights: Option[DVector], p: Broadcast[DenseVector], dual: Double,
    scale: Double = 1.0): DenseVector = {
    val result = matrix.gramianProduct(p, weights).values
    var i = 0
    while (i < result.length) {
      result(i) = scale * result(i) + dual * p.value(i)
      i += 1
    }
    new DenseVector(result)
  }

  /** First nonzero rank of the adaptive preconditioner, after Jacobi misses its target. */
  val DefaultPreconditionerRank: Int = 50

  // Budget for factors and aggregation workspace. Four double arrays per row/rank
  // conservatively cover the factors, column/reduction buffers and allocation headroom.
  // O(m) iteration vectors and Spark runtime overhead are outside this factor budget.
  private[spark_lp] val PreconditionerMemoryBudget: Long = 256L * 1024 * 1024
  private val PreconditionerBytesPerRowRank: Long = 4L * 8L
  private val PreconditionerFlopsBudget: Double = 2e10 // ~seconds of driver factorization time

  /** The largest preconditioner rank the driver budgets allow for `m` constraint rows. */
  private[spark_lp] def autoMaxRank(m: Int, memoryBytes: Long = PreconditionerMemoryBudget): Int = {
    val memoryCap = math.min(Int.MaxValue.toLong, memoryBytes / (PreconditionerBytesPerRowRank * math.max(1, m))).toInt
    val flopsCap = math.sqrt(PreconditionerFlopsBudget / math.max(1, m)).toInt
    math.min(m, math.min(memoryCap, flopsCap))
  }
}
