package com.github.vbmacher.spark_lp

import com.github.vbmacher.spark_lp.newton.{CgConfig, NewtonSolver}

import com.github.vbmacher.spark_lp.dsl.LpNumericalException
import com.github.vbmacher.spark_lp.newton.{NewtonSystem, NewtonSystemFactory}
import com.github.vbmacher.spark_lp.vectors.dense_vector.implicits.DenseVectorOps
import com.github.vbmacher.spark_lp.vectors.dmatrix.implicits._
import com.github.vbmacher.spark_lp.vectors.dvector.implicits._
import com.github.vbmacher.spark_lp.vectors.{DMatrix, DVector}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.SparkException
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.control.NonFatal

object LP extends LazyLogging {

  /** How a solver run terminated. */
  private[spark_lp] sealed trait Termination

  private[spark_lp] object Termination {

    /** All three convergence conditions were met within tolerance. */
    case object Converged extends Termination

    /** The iteration budget was exhausted without convergence or a certificate. */
    case object IterationLimit extends Termination

    /** A cooperative stop was requested; candidate availability is explicit. */
    case object Stopped extends Termination

    /**
      * A Farkas certificate of primal infeasibility was found: the normalized dual iterate
      * `y = lambda / (b^T lambda)` satisfies `A^T y <= infeasibilityTolerance` componentwise with
      * `b^T y = 1`, so no `x >= 0` with `A x = b` exists (within tolerance).
      */
    case object PrimalInfeasible extends Termination

    /**
      * A Farkas certificate of dual infeasibility was found: the normalized primal iterate
      * `z = x / |c^T x|` satisfies `z >= 0`, `c^T z = -1` and
      * `||A z||_inf <= infeasibilityTolerance`, so the dual is infeasible and the primal is
      * unbounded or infeasible (unbounded when a primal-feasible point is known).
      */
    case object DualInfeasible extends Termination
  }

  /**
    * Detailed result of one solver run.
    *
    * @param objectiveValue      final objective value `c^T x` of the returned iterate.
    * @param x                   the returned completed iterate (best feasible on intentional stops), or an
    *                            empty RDD when `candidate.available` is false.
    * @param iterations          the number of completed iterations.
    * @param termination         how the run terminated (see [[Termination]]).
    * @param primalResidual      final `||A x - b|| / (1 + ||b||)`.
    * @param dualResidual        final `||A^T lambda + s - c|| / (1 + ||c||)`.
    * @param dualityGap          final `|c^T x - b^T lambda| / (1 + |b^T lambda)|`.
    * @param primalCertificate   on [[Termination.PrimalInfeasible]], the normalized Farkas ray
    *                            `y = lambda / (b^T lambda)` (so `b^T y = 1`, `A^T y <= eps`).
    * @param dualCertificate     on [[Termination.DualInfeasible]], the normalized Farkas ray
    *                            `z = x / |c^T x|` (so `z >= 0`, `c^T z = -1`, `||A z||_inf <= eps`).
    * @param certificateResidual residual quality of the reported certificate:
    *                            `max(0, max_i (A^T y)_i)` for a primal-infeasibility certificate,
    *                            `||A z||_inf` for a dual-infeasibility one; `NaN` when neither
    *                            certificate was found.
    * @param dualObjectiveValue final dual objective `b^T lambda`; `NaN` when unavailable.
    * @param innerIterations total iterations performed by iterative inner linear solves.
    * @param preconditionerRank highest partial-Cholesky preconditioner rank used.
    * @param stopReason configured policy that stopped the solve, when applicable.
    * @param candidate availability and original-model feasibility of `x`.
    * @param innerRestarts total conjugate-gradient restarts across inner solves.
    */
  private[spark_lp] case class SolveSummary(
    objectiveValue: Double,
    x: DVector,
    iterations: Int,
    termination: Termination,
    primalResidual: Double,
    dualResidual: Double,
    dualityGap: Double,
    primalCertificate: Option[DenseVector] = None,
    dualCertificate: Option[DVector] = None,
    certificateResidual: Double = Double.NaN,
    dualObjectiveValue: Double = Double.NaN,
    innerIterations: Int = 0,
    preconditionerRank: Int = 0,
    stopReason: Option[StopReason] = None,
    candidate: CandidateInfo = CandidateInfo.Unavailable,
    innerRestarts: Int = 0)

  private final case class SolveOptions(
    tolerance: Double,
    maxIter: Int,
    etaIter: Double,
    valueCap: Double,
    eps: Double,
    infeasibilityTolerance: Double,
    solver: NewtonSolver,
    cgTolerance: Double,
    cgMaxIterations: Int,
    stopAfterIteration: Option[Int => Boolean],
    cgConfig: CgConfig,
    control: SolveControl,
    candidateViolation: Option[DVector => Double],
    nanoTime: () => Long,
    inspectConverged: Option[(DVector, DenseVector, DVector) => Unit],
    quadratic: Option[DVector],
    initialPrimal: Option[DVector],
    onStartApplied: () => Unit)

  /**
    * Solves a continuous linear program in equality form:
    * `minimize c^T x` subject to `A x = b` and `x >= 0`.
    *
    * This compact method returns the last candidate directly. The internal `solveSummary` variant
    * needs the termination reason, residuals, or infeasibility certificates.
    *
    * @param c distributed objective coefficients, one value per solver variable.
    * @param AT distributed transpose of `A`; each row contains one variable's coefficients across
    *           all equality constraints.
    * @param b right-hand side of the equality constraints, held on the driver.
    * @param tolerance relative primal residual, dual residual, and duality-gap target.
    * @param maxIter maximum number of interior-point iterations.
    * @param etaIter fraction of the maximum positive step used for each iterate update.
    * @param valueCap upper limit used when computing the historical Cholesky scaling weights.
    * @param eps minimum magnitude used to protect divisions in the interior-point updates.
    * @param infeasibilityTolerance threshold for the Farkas infeasibility certificate tests.
    * @param solver                 how to solve the per-iteration normal-equations systems (see
    *                               [[com.github.vbmacher.spark_lp.newton.NewtonSolver]]). The default
    *                               [[com.github.vbmacher.spark_lp.newton.NewtonSolver.Auto]] uses the
    *                               driver-local Cholesky factorization up to
    *                               [[com.github.vbmacher.spark_lp.newton.NewtonSolver.AutoCholeskyLimit]]
    *                               constraint rows and the
    *                               matrix-free conjugate gradient beyond that.
    * @param cgTolerance            relative residual at which a conjugate-gradient solve is
    *                               accepted (matrix-free solver only).
    * @param cgMaxIterations        CG step limit per normal-equations solve; values < 1 select
    *                               `min(max(100, 2m), 1000)` (matrix-free solver only).
    * @param cgConfig               regularization and preconditioner controls for the
    *                               conjugate-gradient strategy.
    * @param spark                  Spark session that owns the distributed inputs and result.
    * @return objective value and candidate vector returned by the run.
    */
  def solve(
    c: DVector,
    AT: DMatrix,
    b: DenseVector,
    tolerance: Double = 1e-8,
    maxIter: Int = 50,
    etaIter: Double = 0.999,
    valueCap: Double = 1e20,
    eps: Double = 1e-20,
    infeasibilityTolerance: Double = 1e-8,
    solver: NewtonSolver = NewtonSolver.Auto,
    cgTolerance: Double = 1e-10,
    cgMaxIterations: Int = 0,
    cgConfig: CgConfig = CgConfig()
  )(implicit spark: SparkSession): (Double, DVector) = {
    val summary = solveSummary(c, AT, b, tolerance, maxIter, etaIter, valueCap, eps, infeasibilityTolerance,
      solver, cgTolerance, cgMaxIterations, cgConfig = cgConfig)
    (summary.objectiveValue, summary.x)
  }

  /**
    * Solves the same equality-form problem as [[solve]] and returns its full termination record.
    *
    * After each iteration's residual update, two scale-invariant Farkas certificate tests are
    * evaluated against `infeasibilityTolerance` (see [[Termination.PrimalInfeasible]] and
    * [[Termination.DualInfeasible]]); when one holds, the loop stops early and the normalized
    * certificate ray is retained in the summary. Otherwise the explicit iteration budget bounds
    * the run; growing residuals alone do not establish infeasibility.
    *
    * Numerical failures (a non-positive-definite Gramian during initialization or an iteration's
    * normal-equations solve — a Cholesky breakdown or a stalled conjugate-gradient run — or a
    * zero iterate element) are wrapped in
    * [[com.github.vbmacher.spark_lp.dsl.LpNumericalException]] naming the phase and the count of
    * completed iterations — except that a failure during an iteration is first reclassified by
    * running the same certificate tests (at the same `infeasibilityTolerance`) on the last
    * completed iterate: infeasible instances frequently degenerate the Cholesky step before the
    * residual-level certificate crisply forms.
    */
  private[spark_lp] def solveSummary(
    c: DVector,
    AT: DMatrix,
    b: DenseVector,
    tolerance: Double = 1e-8,
    maxIter: Int = 50,
    etaIter: Double = 0.999,
    valueCap: Double = 1e20,
    eps: Double = 1e-20,
    infeasibilityTolerance: Double = 1e-8,
    solver: NewtonSolver = NewtonSolver.Auto,
    cgTolerance: Double = 1e-10,
    cgMaxIterations: Int = 0,
    stopAfterIteration: Option[Int => Boolean] = None,
    cgConfig: CgConfig = CgConfig(),
    control: SolveControl = SolveControl(),
    candidateViolation: Option[DVector => Double] = None,
    nanoTime: () => Long = () => System.nanoTime(),
    inspectConverged: Option[(DVector, DenseVector, DVector) => Unit] = None,
    quadratic: Option[DVector] = None,
    initialPrimal: Option[DVector] = None,
    onStartApplied: () => Unit = () => ()
  )(implicit spark: SparkSession): SolveSummary = solveWithSummary(c, AT, b, SolveOptions(
    tolerance, maxIter, etaIter, valueCap, eps, infeasibilityTolerance, solver, cgTolerance,
    cgMaxIterations, stopAfterIteration, cgConfig, control, candidateViolation, nanoTime,
    inspectConverged, quadratic, initialPrimal, onStartApplied))

  private def solveWithSummary(c: DVector, AT: DMatrix, b: DenseVector, options: SolveOptions)
    (implicit spark: SparkSession): SolveSummary = {
    import options._
    validateParameters(tolerance, maxIter, etaIter, valueCap, eps, infeasibilityTolerance, cgTolerance)
    require(b.size > 0 && b.values.forall(v => !v.isNaN && !v.isInfinite), "b must be nonempty and finite")
    val caches = new CachedRDDs
    val monitor = new SolveMonitor(control, nanoTime)
    try {
      caches.cache(c)
      caches.cache(AT)
      quadratic.foreach(caches.cache(_))
      def gradient(x: DVector): DVector = quadratic.map(q => c.combine(1.0, 1.0, q.entrywiseProd(x))).getOrElse(c)
      // Reuse the wrapper so adjoint products discover the matrix dimensions only once.
      val matrix = new DMatrixOps(AT)
      lazy val certificateScales = certificateScaling(c, AT, b, caches)

      val resolvedSolver = resolveNewtonSolver(solver, b.size)
      val systemFactory: NewtonSystemFactory = resolvedSolver match {
        case NewtonSolver.ConjugateGradient => new newton.CgFactory(cgTolerance, cgMaxIterations, config = cgConfig, monitor = monitor)
        case _ => new newton.CholeskyFactory(monitor)
      }
      logger.info(s"Normal-equations solver: $resolvedSolver, rows=${b.size}, " +
        s"matrixPartitions=${AT.getNumPartitions}, autoCholeskyLimit=${NewtonSolver.AutoCholeskyLimit}")

      // run initialization
      val init =
        try {
          monitor.phase(SolvePhase.Initialization)
          Initialize.init(c, AT, b, systemFactory)
        } catch {
          case stopped: SolveStopped =>
            logger.info(s"LP stopped during initialization: reason=${stopped.reason} candidate=unavailable")
            return SolveSummary(Double.NaN, spark.sparkContext.emptyRDD, 0, Termination.Stopped,
              Double.NaN, Double.NaN, Double.NaN, stopReason = Some(stopped.reason),
              innerIterations = systemFactory.innerIterations, preconditionerRank = systemFactory.maximumRank,
              innerRestarts = systemFactory.innerRestarts)
          case e if isNumericalFailure(e) => throw numericalFailure("initialization", 0, e)
        }

      var x = initialPrimal.map { supplied =>
        val validated = supplied.zip(init.x).map { case (hint, default) =>
          require(hint.size == default.size && hint.values.forall(v => java.lang.Double.isFinite(v) && v > 0.0),
            "Initial primal blocks must match the solver layout and contain finite positive values")
          new DenseVector(hint.values.clone())
        }
        caches.cache(validated); validated.count(); onStartApplied(); validated
      }.getOrElse(init.x)
      caches.cache(x)

      var lambda = init.lambda
      var lambdaBroadcast = spark.sparkContext.broadcast(lambda)

      var s = init.s
      caches.cache(s)

      // set number of unknown in lp
      val unknowns = init.rows

      // set number of equations in lp
      val equations = init.cols

      // initial objective value
      var cTx = Double.PositiveInfinity

      var converged = false
      var earlyTermination: Option[Termination] = None
      var primalCertificate: Option[DenseVector] = None
      var dualCertificate: Option[DVector] = None
      var certificateResidual = Double.NaN
      var iter = 1
      var completedIterations = 0
      var stopReason: Option[StopReason] = None
      var bestCandidate: Option[SolveSummary] = None
      var lastCandidate = CandidateInfo.Unavailable
      val progress = control.stagnation.map(new OuterProgress(_))

      var primalResidual = Double.NaN
      var dualResidual = Double.NaN
      var dualityGap = Double.NaN

      // constants of the residual normalizations
      val normB = math.sqrt(b.dot(b))
      val normC = math.sqrt(c.dot(c))

      // A large unrelated RHS must not relax the accuracy of a small constraint row.
      val newtonAbsTolerance = 0.1 * tolerance * (1.0 + b.values.map(math.abs).min)

      var dLambdaAffBroadcast: Broadcast[DenseVector] = null
      var dLambdaBroadcast: Broadcast[DenseVector] = null

      try {
        while (!converged && earlyTermination.isEmpty && iter <= maxIter) {

          // the last completed iterate, for certificate-based reclassification of numerical failures
          val x0 = x
          val lambda0 = lambda
          val s0 = s

          var newtonSystem: NewtonSystem = null
          val temporary = new CachedRDDs
          try {
            monitor.iteration = iter
            monitor.check()
            // Complementarity belongs to the current iterate, not the starting point.
            val mu = x.dot(s) / unknowns
            // A^T * x - b
            var rb = matrix.adjointProduct(x).combine(1.0, -1.0, b)

            // A * lambda + s - c
            var rc = AT.product(lambdaBroadcast).combine(1.0, 1.0, s.diff(gradient(x)))
            temporary.cache(rc)

            // Proximal references are the current x/lambda. The original residuals are
            // unchanged; only the Newton equations contain Rp and Rd.
            val effectiveSlack = quadratic.map(q => s.combine(1.0, 1.0, q.entrywiseProd(x))).getOrElse(s)
            // QP free-variable splits need the configured proximal stabilization in their nullspace.
            val rho = if (quadratic.isEmpty && systemFactory.primalRegularization > 0.0)
              systemFactory.primalRegularization * math.min(1.0,
                effectiveSlack.zipElements(x, (si, xi) => si / xi).minValue)
            else systemFactory.primalRegularization
            val weights = if (resolvedSolver == NewtonSolver.ConjugateGradient || quadratic.nonEmpty) regularizedWeights(x, effectiveSlack, rho)
              else normalEquationWeights(x, s, eps, valueCap)
            val D2 = temporary.cache(weights.squared)
            newtonSystem = systemFactory.build(AT, equations, Some(weights))

            // qAff = -X S e, hAff = rc + qAff / x = rc - s.
            val hAff = rc.diff(s)
            val dLambdaAffRightSide = rb.combine(-1.0, -1.0, matrix.adjointProduct(D2.entrywiseProd(hAff)))
            val dLambdaAff = newtonSystem.solve(dLambdaAffRightSide, newtonAbsTolerance)
            dLambdaAffBroadcast = rebroadcast(dLambdaAffBroadcast, dLambdaAff)
            val (dxAff0, dsAff0) = weights.recoverDirections(
              hAff, rc, AT.product(dLambdaAffBroadcast), rho)
            val dxAff = temporary.cache(dxAff0)
            val dsAff = temporary.cache(quadratic.map(q => dsAff0.combine(1.0, 1.0, q.entrywiseProd(dxAff))).getOrElse(dsAff0))

            // Calculate following Doubles alphaPriAff, alphaDualAff, muAff (14.32), (14.33)
            val alphaPriAff = math.min(1.0, x.entrywiseNegDiv(dxAff).minValue)
            val alphaDualAff = math.min(1.0, s.entrywiseNegDiv(dsAff).minValue)
            val muAff = {
              val nx = x.combine(1.0, alphaPriAff, dxAff)
              val ns = s.combine(1.0, alphaDualAff, dsAff)
              nx.dot(ns) / unknowns
            }

            val sigma = math.min(1.0, math.pow(muAff / mu, 3)) // heuristic

            // q = -X S e - dxAff*dsAff + sigma*mu*e. Eliminate with the same W
            // used by the operator, then recover both directions from the regularized KKT.
            val xInv = x.mapElements { a =>
              require(a > 0.0 && !a.isInfinite, "Newton iterate X must be finite and positive")
              1.0 / a
            }
            val h = hAff.diff(xInv.entrywiseProd(dxAff.entrywiseProd(dsAff)))
              .combine(1.0, sigma * mu, xInv)
            val dLambdaRightSide = rb.combine(-1.0, -1.0, matrix.adjointProduct(D2.entrywiseProd(h)))
            val dLambda = newtonSystem.solve(dLambdaRightSide, newtonAbsTolerance)
            dLambdaBroadcast = rebroadcast(dLambdaBroadcast, dLambda)
            val (dx0, ds0) = weights.recoverDirections(
              h, rc, AT.product(dLambdaBroadcast), rho)
            val dx = temporary.cache(dx0)
            val ds = temporary.cache(quadratic.map(q => ds0.combine(1.0, 1.0, q.entrywiseProd(dx))).getOrElse(ds0))

            val alphaPrimalIterMax = x.entrywiseNegDiv(dx).minValue
            val alphaDualIterMax = s.entrywiseNegDiv(ds).minValue
            val alphaPrimalIter = math.min(1.0, etaIter * alphaPrimalIterMax)
            val alphaDualIter = math.min(1.0, etaIter * alphaDualIterMax)

            monitor.check()
            // x = x + alphaPriIter * dx
            x = caches.checkpoint(x.combine(1.0, alphaPrimalIter, dx))

            // lambda = lambda + alphaDualIter * dLambda
            lambda = new DenseVector((lambdaBroadcast.value.toBreeze + alphaDualIter * dLambda.toBreeze).toArray)
            lambdaBroadcast = rebroadcast(lambdaBroadcast, lambda)

            // s = s + alphaDualIter * ds
            s = caches.checkpoint(s.combine(1.0, alphaDualIter, ds))

            rb = matrix.adjointProduct(x).combine(1.0, -1.0, b)
            rc = temporary.cache(AT.product(lambdaBroadcast).combine(1.0, 1.0, s.diff(gradient(x))))
            val objectiveAndMin = c.zipPartitions(x) { (costs, values) =>
              val cv = costs.next().values
              val xv = values.next().values
              var objective = 0.0
              var minimum = Double.PositiveInfinity
              var i = 0
              while (i < xv.length) {
                objective += cv(i) * xv(i)
                minimum = math.min(minimum, xv(i))
                i += 1
              }
              Iterator.single((objective, minimum))
            }.reduce { case ((a, amin), (b, bmin)) => (a + b, math.min(amin, bmin)) }
            val xQx = quadratic.map(q => x.dot(q.entrywiseProd(x))).getOrElse(0.0)
            cTx = objectiveAndMin._1 + 0.5 * xQx

            val bTlambda = b.dot(lambda) - 0.5 * xQx
            val normRb = math.sqrt(rb.dot(rb))
            val normRc = math.sqrt(rc.dot(rc))
            val covg1 = normRb / (1 + normB)
            val covg2 = normRc / (1 + normC)
            val covg3 = math.abs(cTx - bTlambda) / (1 + math.abs(bTlambda))

            primalResidual = covg1
            dualResidual = covg2
            dualityGap = covg3

            val rowResidual = rb.values.zip(b.values).map { case (r, rhs) =>
              math.abs(r) / (1.0 + math.abs(rhs))
            }.max
            converged = (covg1 < tolerance) && (rowResidual < tolerance) &&
              (covg2 < tolerance) && (covg3 < tolerance)

            if (!converged) {
              certificatesOnIterate(c, AT, b, x, lambda, infeasibilityTolerance, quadratic, certificateScales).foreach { cert =>
                earlyTermination = Some(cert.termination)
                primalCertificate = cert.primalCertificate
                dualCertificate = cert.dualCertificate
                certificateResidual = cert.residual
              }
            }
            logger.debug(s"LP iteration=$iter sigma=$sigma primalResidual=$covg1 " +
              s"dualResidual=$covg2 gap=$covg3 converged=$converged cTx=$cTx bTlambda=$bTlambda")

            completedIterations = iter
            val violation = candidateViolation.map(_(x)).getOrElse {
              val rowViolation = rb.values.zip(b.values).map { case (r, rhs) =>
                math.abs(r) / (1.0 + math.abs(rhs))
              }.max
              math.max(rowViolation, math.max(0.0, -objectiveAndMin._2))
            }
            val feasible = !cTx.isNaN && !cTx.isInfinite && violation <= control.feasibilityTolerance
            lastCandidate = CandidateInfo(available = true, feasible = feasible, iteration = Some(iter))
            if (feasible && bestCandidate.forall(cTx < _.objectiveValue)) {
              bestCandidate.foreach { previous => if (previous.x ne x0) caches.release(previous.x) }
              bestCandidate = Some(SolveSummary(cTx, x, iter, Termination.Stopped, covg1, covg2, covg3,
                dualObjectiveValue = bTlambda, candidate = lastCandidate))
            }
            // Both replacements have been materialised and checkpointed by the residual actions.
            if (!bestCandidate.exists(_.x eq x0)) caches.release(x0)
            caches.release(s0)
            val terminal = converged || earlyTermination.nonEmpty
            monitor.report(SolvePhase.OuterIteration,
              iterate = Some(IterationProgress(cTx, covg1, covg2, covg3, feasible)), terminal = terminal)
            if (!terminal) {
              if (monitor.callback(stopAfterIteration.exists(_(iter)))) throw SolveStopped(StopReason.UserRequested)
              if (progress.exists(_.observe(iter, feasible, cTx, violation, covg1, covg2, covg3)))
                throw SolveStopped(StopReason.NoProgress)
            }

          } catch {
            case stopped: SolveStopped =>
              earlyTermination = Some(Termination.Stopped)
              stopReason = Some(stopped.reason)
              // Safe checks run before updates or after a fully completed iteration.
            case e if isNumericalFailure(e) =>
              // Before wrapping into LpNumericalException, test the last completed iterate for a
              // Farkas certificate (at the same public infeasibilityTolerance) that explains the
              // degeneration. Without a certificate, report a numerical failure.
              val certificate =
                try certificatesOnIterate(c, AT, b, x0, lambda0, infeasibilityTolerance, quadratic, certificateScales)
                catch {
                  case NonFatal(_) => None
                }
              certificate match {
                case Some(cert) =>
                  earlyTermination = Some(cert.termination)
                  primalCertificate = cert.primalCertificate
                  dualCertificate = cert.dualCertificate
                  certificateResidual = cert.residual
                  // report the last completed iterate; the failed iteration may have partially updated state
                  if (x ne x0) caches.release(x)
                  if (s ne s0) caches.release(s)
                  x = x0
                  lambda = lambda0
                  s = s0
                  cTx = cert.cTx + 0.5 * quadratic.map(q => x.dot(q.entrywiseProd(x))).getOrElse(0.0)
                  lastCandidate = CandidateInfo(true, false, Some(completedIterations))
                  logger.info(s"Numerical failure in iteration $iter reclassified as ${cert.termination} " +
                    s"(certificate residual ${cert.residual})")
                case None => throw numericalFailure(s"iteration $iter", iter - 1, e)
              }
          } finally {
            if (newtonSystem != null) newtonSystem.release()
            temporary.close()
          }
          iter += 1
        }

      } finally {
        if (dLambdaAffBroadcast != null) dLambdaAffBroadcast.unpersist(blocking = false)
        if (dLambdaBroadcast != null) dLambdaBroadcast.unpersist(blocking = false)
        lambdaBroadcast.unpersist(blocking = false)
      }

      val termination =
        if (converged) Termination.Converged
        else earlyTermination.getOrElse(Termination.IterationLimit)

      // Internal validation hook: inspect the actual primal/dual iterate before releasing its
      // caches. Only converged results qualify; stopped runs may return an earlier candidate.
      if (converged) monitor.callback(inspectConverged.foreach(_(x, lambda, s)))

      val last = SolveSummary(
        objectiveValue = if (!lastCandidate.available) Double.NaN else cTx,
        x = if (!lastCandidate.available) spark.sparkContext.emptyRDD[DenseVector] else x,
        iterations = completedIterations,
        termination = termination,
        primalResidual = primalResidual,
        dualResidual = dualResidual,
        dualityGap = dualityGap,
        primalCertificate = primalCertificate,
        dualCertificate = dualCertificate,
        certificateResidual = certificateResidual,
        dualObjectiveValue = b.dot(lambda) - 0.5 * quadratic.map(q => x.dot(q.entrywiseProd(x))).getOrElse(0.0),
        innerIterations = systemFactory.innerIterations,
        preconditionerRank = systemFactory.maximumRank,
        candidate = lastCandidate)
      val intentional = termination == Termination.Stopped || termination == Termination.IterationLimit
      val selected = if (intentional) bestCandidate.getOrElse(last) else last
      caches.keep(selected.x)
      val result = selected.copy(iterations = completedIterations, termination = termination,
        primalCertificate = primalCertificate, dualCertificate = dualCertificate,
        certificateResidual = certificateResidual, innerIterations = systemFactory.innerIterations,
        preconditionerRank = systemFactory.maximumRank,
        innerRestarts = systemFactory.innerRestarts,
        stopReason = if (termination == Termination.IterationLimit) Some(StopReason.IterationLimit) else stopReason)
      logger.info(s"LP finished: solver=$resolvedSolver termination=$termination stopReason=${result.stopReason} " +
        s"iterations=$completedIterations candidate=${result.candidate} objective=${result.objectiveValue} " +
        s"primalResidual=${result.primalResidual} dualResidual=${result.dualResidual} gap=${result.dualityGap}")
      result
    } catch {
      case failed: CallbackFailed => throw failed.error
    } finally caches.close()
  }

  private[spark_lp] def validateParameters(
    tolerance: Double, maxIter: Int, etaIter: Double, valueCap: Double, eps: Double,
    infeasibilityTolerance: Double, cgTolerance: Double): Unit = {
    Seq("tolerance" -> tolerance, "valueCap" -> valueCap, "epsilon" -> eps,
      "infeasibilityTolerance" -> infeasibilityTolerance, "cgTolerance" -> cgTolerance).foreach {
      case (name, value) => require(value > 0.0 && !value.isInfinite, s"$name must be finite and positive")
    }
    require(maxIter > 0, "maxIterations must be positive")
    require(etaIter > 0.0 && etaIter < 1.0, "etaIteration must be between 0 and 1 (exclusive)")
  }

  /**
    * Certificate accepted for a completed iterate.
    *
    * @param termination certificate type established by the iterate.
    * @param primalCertificate normalized equality-multiplier ray for primal infeasibility.
    * @param dualCertificate normalized primal-variable ray for dual infeasibility.
    * @param residual maximum equation or sign violation of the retained ray.
    * @param cTx objective product `c^T x` before ray normalization.
    */
  private case class Certificate(
    termination: Termination,
    primalCertificate: Option[DenseVector],
    dualCertificate: Option[DVector],
    residual: Double,
    cTx: Double)

  private case class CertificateScaling(
    rows: Array[Double], columns: DVector, primalColumnNorms: DVector,
    dualRowNorms: Array[Double], rhsNorm: Double, costNorm: Double)

  /** Equilibrate certificate coordinates without changing the model or the solver iterates. */
  private def certificateScaling(c: DVector, AT: DMatrix, b: DenseVector,
    caches: CachedRDDs): CertificateScaling = {
    def add(left: Array[Double], right: Array[Double]): Array[Double] = {
      var j = 0
      while (j < left.length) { left(j) += right(j); j += 1 }
      left
    }
    val rows = AT.treeAggregate(new Array[Double](b.size))(
      (norms, row) => { row.foreachActive((j, a) => norms(j) += math.abs(a)); norms }, add)
    rows.indices.foreach { j =>
      rows(j) = math.max(rows(j), math.abs(b(j)))
      if (rows(j) == 0.0) rows(j) = 1.0
    }
    val primalColumns = caches.cache(AT.mapPartitions(partition => Iterator.single(new DenseVector(
      partition.map { row =>
        var norm = 0.0
        row.foreachActive((j, a) => norm += math.abs(a) / rows(j))
        norm
      }.toArray))))
    val columnNorms = AT.mapPartitions(partition => Iterator.single(new DenseVector(
      partition.map { row => var norm = 0.0; row.foreachActive((_, a) => norm += math.abs(a)); norm }.toArray)))
    val columns = caches.cache(c.zipElements(columnNorms, (cost, norm) => {
      val scale = math.max(math.abs(cost), norm)
      if (scale == 0.0) 1.0 else scale
    }))
    val dualRows = AT.zipPartitions(columns) { (partition, scales) =>
      val values = scales.next().values
      val norms = new Array[Double](b.size)
      var i = 0
      partition.foreach { row =>
        val scale = values(i)
        row.foreachActive((j, a) => norms(j) += math.abs(a) / scale)
        i += 1
      }
      Iterator.single(norms)
    }.treeAggregate(new Array[Double](b.size))(add, add)
    CertificateScaling(rows, columns, primalColumns, dualRows,
      b.values.indices.map(j => math.abs(b(j)) / rows(j)).sum,
      c.zipElements(columns, (cost, scale) => math.abs(cost) / scale).sum())
  }

  /**
    * Runs the two Farkas certificate tests on a completed iterate `(x, lambda)`, at the same
    * `infeasibilityTolerance` used inside the iteration loop.
    */
  private def certificatesOnIterate(
    c: DVector,
    AT: DMatrix,
    b: DenseVector,
    x: DVector,
    lambda: DenseVector,
    infeasibilityTolerance: Double, quadratic: Option[DVector], scales: CertificateScaling): Option[Certificate] = {
    val cTx = c.dot(x)
    val bTlambda = b.dot(lambda)

    val lambdaNorm = lambda.values.indices.map(j => math.abs(lambda(j)) * scales.rows(j)).max
    val primalMargin = Numerics.relativeResidual(bTlambda,
      scales.rhsNorm * lambdaNorm)
    val primal = if (bTlambda > 0 && Numerics.isFinite(primalMargin) && primalMargin > infeasibilityTolerance) {
      val ray = new DenseVector(lambda.values.map(_ / bTlambda))
      val activity = AT.product(ray)
      val quality = math.max(0.0, activity.maxValue)
      val relative = activity.zipElements(scales.primalColumnNorms, (a, norm) =>
        Numerics.relativeResidual(math.max(0.0, a), norm * (lambdaNorm / bTlambda))).maxValue
      if (quality <= infeasibilityTolerance && relative <= infeasibilityTolerance) {
        Some(Certificate(
          Termination.PrimalInfeasible,
          primalCertificate = Some(ray),
          dualCertificate = None,
          residual = quality,
          cTx = cTx))
      } else None
    } else None

    primal.orElse {
      if (cTx < 0) {
        val scaledXNorm = x.entrywiseProd(scales.columns).mapElements(math.abs).maxValue
        val margin = Numerics.relativeResidual(-cTx, scales.costNorm * scaledXNorm)
        if (!Numerics.isFinite(margin) || margin <= infeasibilityTolerance) return None
        val cTxAbs = math.abs(cTx)
        val ray = x.mapElements(_ / cTxAbs)
        val ax = AT.adjointProduct(ray).values
        val relative = ax.zip(scales.dualRowNorms).map { case (a, norm) =>
          Numerics.relativeResidual(math.abs(a), norm * (scaledXNorm / cTxAbs))
        }.max
        val qInf = quadratic.map(q => q.entrywiseProd(x).maxValue).getOrElse(0.0)
        val curvature = quadratic.map(q => q.zipElements(x, (qi, xi) =>
          if (qi == 0.0) 0.0 else math.abs(xi)).entrywiseProd(scales.columns)
          .mapElements(_ / scaledXNorm).maxValue).getOrElse(0.0)
        val quality = math.max(ax.map(math.abs).max, qInf / cTxAbs)
        if (quality <= infeasibilityTolerance && relative <= infeasibilityTolerance && curvature <= infeasibilityTolerance) {
          Some(Certificate(
            Termination.DualInfeasible,
            primalCertificate = None,
            dualCertificate = Some(ray),
            residual = quality,
            cTx = cTx))
        } else None
      } else None
    }
  }

  /** Failure modes of the linear algebra underneath the solver, possibly wrapped by Spark. */
  private def isNumericalFailure(e: Throwable): Boolean = e match {
    case _: MatchError | _: IllegalArgumentException | _: IllegalStateException | _: AssertionError => true
    case e: SparkException => e.getCause != null && isNumericalFailure(e.getCause)
    case _ => false
  }

  private def numericalFailure(phase: String, completedIterations: Int, cause: Throwable): LpNumericalException = {
    val detail = Option(cause.getMessage).getOrElse(cause.toString)
    new LpNumericalException(
      phase = phase,
      completedIterations = completedIterations,
      message = s"Numerical failure during $phase (completed iterations: $completedIterations): $detail. " +
        "No infeasibility or unboundedness certificate was established. Check scaling, " +
        "strictly positive iterates and the selected Newton solver settings; Cholesky requires full row rank.",
      cause = cause)
  }

  def rebroadcast[T: ClassTag](old: Broadcast[T], v: T)(implicit spark: SparkSession): Broadcast[T] = {
    if (old != null) old.unpersist(blocking = false)
    spark.sparkContext.broadcast(v)
  }

  /** Resolves Auto from the already-local equality-form row count. */
  private[spark_lp] def resolveNewtonSolver(solver: NewtonSolver, equations: Int): NewtonSolver =
    NewtonSolver.resolve(solver, equations)

  /** W = (S/X + Rp)^(-1), evaluated without forming the potentially overflowing X/S.
    * No inverse-slack cap: the same diagonal participates in the operator and recovery.
    */
  private[spark_lp] def regularizedWeights(x: DVector, s: DVector, rho: Double): newton.Weights = {
    val squared = x.zip(s).map { case (xp, sp) =>
      require(xp.size == sp.size, "Newton vectors must have matching partitions")
      new DenseVector(xp.values.zip(sp.values).map { case (xi, si) =>
        require(xi > 0.0 && si > 0.0 && !xi.isInfinite && !si.isInfinite,
          "Newton iterates must be finite and positive")
        val w = 1.0 / (si / xi + rho)
        require(w > 0.0 && !w.isInfinite, "Non-finite or underflowed Newton weight")
        w
      })
    }
    newton.Weights(squared.mapElements(math.sqrt), squared)
  }

  /** Historical capped weights for the unregularized Cholesky reference. */
  private[spark_lp] def normalEquationWeights(
    x: DVector,
    s: DVector,
    eps: Double,
    valueCap: Double): newton.Weights = {
    val squared = x.entrywiseProd(
      s.mapElements {
        case a if math.abs(a) < eps => math.signum(a) * valueCap
        case a if math.abs(a) >= eps => math.pow(a, -1)
      }
    ).mapElements {
      case a if a > 0.0 && !a.isInfinite => a
      case a => throw new IllegalArgumentException(s"Found non-positive or non-finite D^2 element: $a")
    }
    newton.Weights(sqrt = squared.mapElements(math.sqrt), squared = squared)
  }
}

/** Owns only caches created by this operation; never releases a caller's persisted inputs. */
private[spark_lp] final class CachedRDDs extends AutoCloseable {
  private val owned = mutable.Set.empty[RDD[_]]

  def cache[T](rdd: RDD[T]): RDD[T] = {
    if (rdd.getStorageLevel == StorageLevel.NONE) {
      rdd.cache()
      owned += rdd
    }
    rdd
  }

  def checkpoint[T](rdd: RDD[T]): RDD[T] = {
    cache(rdd)
    rdd.checkpoint()
    rdd
  }

  def keep[T](rdd: RDD[T]): RDD[T] = { owned -= rdd; rdd }

  def release(rdd: RDD[_]): Unit = {
    if (owned.remove(rdd)) rdd.unpersist(blocking = false)
  }

  override def close(): Unit = owned.toVector.foreach(release)
}
