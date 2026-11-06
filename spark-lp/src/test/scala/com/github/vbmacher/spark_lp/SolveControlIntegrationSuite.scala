package com.github.vbmacher.spark_lp

import com.github.vbmacher.spark_lp.newton.{CgConfig, NewtonSolver}

import com.github.vbmacher.spark_lp.vectors.{DMatrix, DVector}
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration._

class SolveControlIntegrationSuite extends AnyFunSuite with DataFrameSuiteBase {
  private def cachedEqualityInputs(
    costs: Seq[Double],
    rows: Seq[org.apache.spark.mllib.linalg.Vector],
    rhs: Array[Double]
  ): (DVector, DMatrix, DenseVector) = {
    val c = sc.parallelize(costs, 2).glom().map(new DenseVector(_)).cache()
    val at = sc.parallelize(rows, 2).cache()
    (c, at, new DenseVector(rhs))
  }

  test("both backends report common setup and inner phases with coherent metric groups") {
    implicit val ss: SparkSession = spark
    for (solver <- Seq(NewtonSolver.Cholesky, NewtonSolver.ConjugateGradient)) {
      val (c, at, b) = cachedEqualityInputs(
      costs = Seq(0.0, 0.0),
      rows = Seq(Vectors.dense(1.0), Vectors.dense(1.0)),
      rhs = Array(1.0))
      val events = ArrayBuffer.empty[SolveProgress]

      val result = LP.solveSummary(c, at, b, solver = solver,
        control = SolveControl(onProgress = events += _))
      try {
        assert(result.termination == LP.Termination.Converged)
        assert(events.exists(_.phase == SolvePhase.SystemSetup))
        assert(events.exists(_.phase == SolvePhase.InnerSolve))
        assert(events.filter(_.phase == SolvePhase.OuterIteration).forall(e => e.iterate.nonEmpty && e.work.isEmpty))
        assert(events.filter(_.phase != SolvePhase.OuterIteration).forall(_.iterate.isEmpty))
        assert(events.filter(_.work.nonEmpty).forall(e =>
          e.phase == SolvePhase.SystemSetup || e.phase == SolvePhase.InnerSolve))

        assert(events.map(_.elapsedSeconds).sliding(2).forall(pair => pair.head <= pair.last))
      } finally { result.x.unpersist(); c.unpersist(); at.unpersist() }
    }
  }

  test("all intentional limits retain an earlier feasible iterate and matching metadata") {
    implicit val ss: SparkSession = spark
    val (c, at, b) = cachedEqualityInputs(
      costs = Seq(0.0, 0.0),
      rows = Seq(Vectors.dense(1.0), Vectors.dense(1.0)),
      rhs = Array(1.0))
    val baseline = sc.getPersistentRDDs.keySet

    val first = LP.solveSummary(c, at, b, maxIter = 1,
      control = SolveControl(feasibilityTolerance = 1e-2))
    val events = ArrayBuffer.empty[SolveProgress]

    val limited = LP.solveSummary(c, at, b, tolerance = 1e-30, maxIter = 4,
      control = SolveControl(feasibilityTolerance = 1e-2, onProgress = events += _))

    val stopped = LP.solveSummary(c, at, b, tolerance = 1e-30,
      stopAfterIteration = Some(_ >= 4), control = SolveControl(feasibilityTolerance = 1e-2))
    var now = 0L

    val timed = LP.solveSummary(c, at, b, tolerance = 1e-30, nanoTime = () => now,
      control = SolveControl(feasibilityTolerance = 1e-2, timeLimit = Some(1.second),
        onProgress = e => if (e.phase == SolvePhase.OuterIteration && e.iteration == 4)
          now = 1000000000L))

    val stalled = LP.solveSummary(c, at, b, tolerance = 1e-30,
      control = SolveControl(feasibilityTolerance = 1e-2,
        stagnation = Some(StagnationConfig(patience = 3, absoluteImprovement = 100.0))))
    try {
      assert(limited.iterations == 4)
      assert(stopped.iterations == 4)
      assert(limited.stopReason.contains(StopReason.IterationLimit))
      assert(stopped.stopReason.contains(StopReason.UserRequested))
      assert(timed.stopReason.contains(StopReason.TimeLimit))
      assert(stalled.stopReason.contains(StopReason.NoProgress))
      Seq(limited, stopped, timed, stalled).foreach { result =>
        assert(result.candidate == CandidateInfo(true, true, Some(1)))
        assert(result.objectiveValue == first.objectiveValue)
        assert(result.primalResidual == first.primalResidual)
        assert(result.dualResidual == first.dualResidual)
        assert(result.dualityGap == first.dualityGap)
        assert(result.x.collect().flatMap(_.values).sameElements(first.x.collect().flatMap(_.values)))
      }

      assert(events.count(_.phase == SolvePhase.OuterIteration) == 4)
      assert(events.filter(_.phase == SolvePhase.OuterIteration).forall(_.iterate.exists(_.feasible)))
    } finally {
      Seq(first, limited, stopped, timed, stalled).foreach(_.x.unpersist(blocking = true))
    }

    assert(sc.getPersistentRDDs.keySet == baseline)
    assert(c.count() == 2)
    assert(at.count() == 2)
    c.unpersist(); at.unpersist()
  }

  test("initialization and in-flight CG stopping expose no partial iterate") {
    implicit val ss: SparkSession = spark
    val (c, at, b) = cachedEqualityInputs(
      costs = Seq(0.0, 0.0),
      rows = Seq(Vectors.dense(1.0), Vectors.dense(1.0)),
      rhs = Array(1.0))
    val baseline = sc.getPersistentRDDs.keySet
    Seq(SolvePhase.Initialization, SolvePhase.InnerSolve, SolvePhase.SystemSetup).foreach { phase =>
      var stop = false

      val result = LP.solveSummary(c, at, b, solver = NewtonSolver.ConjugateGradient,
        cgConfig = CgConfig(preconditionerRank = 1),
        control = SolveControl(onProgress = e => if (e.phase == phase) stop = true,
          shouldStop = () => stop))

      assert(result.termination == LP.Termination.Stopped)
      assert(result.candidate == CandidateInfo.Unavailable)
      assert(result.iterations == 0)
      assert(result.objectiveValue.isNaN)
      assert(result.x.isEmpty())
      assert(sc.getPersistentRDDs.keySet == baseline)
    }
    c.unpersist(); at.unpersist()

    assert(spark.range(3).count() == 3)
  }

  test("a stop after a Cholesky block cleans up without completing initialization") {
    implicit val ss: SparkSession = spark
    val c = sc.parallelize(Seq(1.0, 1.0), 2).glom().map(new DenseVector(_))
    val at = sc.parallelize(Seq(Vectors.dense(1.0, 0.0), Vectors.dense(0.0, 1.0)), 2)
    var stop = false

    val result = LP.solveSummary(c, at, new DenseVector(Array(1.0, 2.0)),
      control = SolveControl(onProgress = e => if (e.phase == SolvePhase.SystemSetup && e.work.exists(_.completed == 1)) stop = true,
        shouldStop = () => stop))

    assert(stop)
    assert(result.candidate == CandidateInfo.Unavailable)
    assert(result.stopReason.contains(StopReason.UserRequested))
  }

  test("stops are checked after actual preconditioner pivots and CG steps") {
    implicit val ss: SparkSession = spark
    for (duringPivot <- Seq(true, false)) {
      val (c, at, b) = cachedEqualityInputs(
      costs = Seq(0.0, 0.0),
      rows = Seq(Vectors.dense(1.0), Vectors.dense(1.0)),
      rhs = Array(1.0))
      var stop = false

      val result = LP.solveSummary(c, at, b, solver = NewtonSolver.ConjugateGradient,
        cgConfig = CgConfig(preconditionerRank = 1),
        control = SolveControl(shouldStop = () => stop, onProgress = { e =>
          if ((duringPivot && e.phase == SolvePhase.SystemSetup && e.work.flatMap(_.preconditionerRank).contains(1)) ||
              (!duringPivot && e.phase == SolvePhase.InnerSolve && e.work.exists(_.completed == 1))) stop = true
        }))

      assert(stop)
      assert(!result.candidate.available)
      assert(result.stopReason.contains(StopReason.UserRequested))
      c.unpersist(); at.unpersist()
    }
  }

  test("inner work cap stops on a true residual without returning an unfinished direction") {
    implicit val ss: SparkSession = spark
    val c = sc.parallelize(Seq(1.0, 2.0, 3.0), 1).glom().map(new DenseVector(_))
    val at = sc.parallelize(Seq(Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 1.0),
      Vectors.dense(0.0, 2.0)), 1)
    val events = ArrayBuffer.empty[SolveProgress]

    val result = LP.solveSummary(c, at, new DenseVector(Array(1.0, 3.0)),
      solver = NewtonSolver.ConjugateGradient,
      control = SolveControl(onProgress = events += _,
        stagnation = Some(StagnationConfig(maxInnerSteps = 1))))

    assert(result.stopReason.contains(StopReason.NoProgress))
    assert(result.candidate == CandidateInfo.Unavailable)
    assert(events.last.work.exists(_.trueResidual))
    assert(events.last.work.map(_.completed).contains(1))
  }

  test("true-residual probes preserve CG directions on a solve longer than 25 steps") {
    implicit val ss: SparkSession = spark
    val m = 40
    val matrix = sc.parallelize(0 to m, 2).map { i =>
      val entries = if (i == 0) Seq(0 -> 1.0)
        else if (i == m) Seq((m - 1) -> -1.0)
        else Seq((i - 1) -> -1.0, i -> 1.0)
      Vectors.sparse(m, entries)
    }.cache()
    val rhs = new DenseVector(Array.tabulate(m)(i => if (i == 0) 1.0 else 0.0))
    val events = ArrayBuffer.empty[SolveProgress]
    val config = CgConfig(preconditionerMemoryBytes = 0)
    val baseline = new newton.CgFactory(1e-10, 100, config)
    val monitored = new newton.CgFactory(1e-10, 100, config,
      new SolveMonitor(SolveControl(onProgress = events += _, stagnation = Some(StagnationConfig()))))
    val first = baseline.build(matrix, m, None)
    val second = monitored.build(matrix, m, None)
    try {
      val expected = first.solve(rhs).values
      val actual = second.solve(rhs).values

      assert(actual.zip(expected).forall { case (a, b) => math.abs(a - b) < 1e-10 })
      assert(monitored.innerIterations == baseline.innerIterations)
      assert(events.exists(e => e.work.exists(_.trueResidual) && e.work.map(_.completed).contains(25)))
      assert(events.last.work.exists(_.trueResidual))
      assert(events.last.work.flatMap(_.residual).exists(_ < 1e-10))
    } finally { first.release(); second.release(); matrix.unpersist() }
  }

  test("callback exceptions propagate unchanged and release solver-owned caches") {
    implicit val ss: SparkSession = spark
    val (c, at, b) = cachedEqualityInputs(
      costs = Seq(0.0, 0.0),
      rows = Seq(Vectors.dense(1.0), Vectors.dense(1.0)),
      rhs = Array(1.0))
    val baseline = sc.getPersistentRDDs.keySet
    val failure = new IllegalStateException("application callback failed")
    val thrown = intercept[IllegalStateException] {
      LP.solveSummary(c, at, b, control = SolveControl(onProgress = { e =>
        if (e.phase == SolvePhase.OuterIteration) throw failure
      }))
    }

    assert(thrown eq failure)
    assert(sc.getPersistentRDDs.keySet == baseline)
    c.unpersist(); at.unpersist()
  }

  test("a completed but infeasible iterate remains available only for diagnostics") {
    implicit val ss: SparkSession = spark
    val c = sc.parallelize(Seq(1.0, 0.0, 0.0), 1).glom().map(new DenseVector(_))
    val at = sc.parallelize(Seq(Vectors.dense(1.0, 1.0), Vectors.dense(1.0, 0.0),
      Vectors.dense(0.0, -1.0)), 1)

    val result = LP.solveSummary(c, at, new DenseVector(Array(1.0, 2.0)),
      stopAfterIteration = Some(_ => true))
    try {
      assert(result.termination == LP.Termination.Stopped)
      assert(result.candidate == CandidateInfo(true, false, Some(1)))
      assert(!result.x.isEmpty())
      assert(!result.objectiveValue.isNaN)
    } finally result.x.unpersist()
  }

  test("stopping in the next Newton solve preserves the last completed candidate") {
    implicit val ss: SparkSession = spark
    val (c, at, b) = cachedEqualityInputs(
      costs = Seq(0.0, 0.0),
      rows = Seq(Vectors.dense(1.0), Vectors.dense(1.0)),
      rhs = Array(1.0))
    var stop = false

    val result = LP.solveSummary(c, at, b, solver = NewtonSolver.ConjugateGradient,
      tolerance = 1e-30, control = SolveControl(feasibilityTolerance = 1e-2,
        onProgress = e => if (e.iteration == 2 && e.phase == SolvePhase.InnerSolve) stop = true,
        shouldStop = () => stop))
    try {
      assert(result.stopReason.contains(StopReason.UserRequested))
      assert(result.iterations == 1)
      assert(result.candidate == CandidateInfo(true, true, Some(1)))
      assert(result.x.collect().flatMap(_.values).forall(v => v >= 0.0 && !v.isNaN))
    } finally { result.x.unpersist(); c.unpersist(); at.unpersist() }
  }

  test("a deadline during initialization returns TimeLimit without a candidate") {
    implicit val ss: SparkSession = spark
    val (c, at, b) = cachedEqualityInputs(
      costs = Seq(0.0, 0.0),
      rows = Seq(Vectors.dense(1.0), Vectors.dense(1.0)),
      rhs = Array(1.0))
    var now = 0L

    val result = LP.solveSummary(c, at, b, nanoTime = () => now,
      control = SolveControl(timeLimit = Some(1.second), onProgress = _ => now = 1000000000L))

    assert(result.stopReason.contains(StopReason.TimeLimit))
    assert(!result.candidate.available)
    c.unpersist(); at.unpersist()
  }

  test("terminal iteration is reported and convergence takes precedence over a stop") {
    implicit val ss: SparkSession = spark
    val (c, at, b) = cachedEqualityInputs(
      costs = Seq(0.0, 0.0),
      rows = Seq(Vectors.dense(1.0), Vectors.dense(1.0)),
      rhs = Array(1.0))
    val events = ArrayBuffer.empty[SolveProgress]
    var stop = false

    val result = LP.solveSummary(c, at, b, tolerance = 1.0,
      control = SolveControl(onProgress = { e =>
        events += e
        if (e.phase == SolvePhase.OuterIteration) stop = true
      }, shouldStop = () => stop))

    assert(result.termination == LP.Termination.Converged)
    assert(result.stopReason.isEmpty)
    assert(events.last.iteration == result.iterations)
    assert(events.last.iterate.map(_.objectiveValue).contains(result.objectiveValue))
    result.x.unpersist(); c.unpersist(); at.unpersist()
  }
}
