package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp._
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration._

class LpDslProgressSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("progress restores sense and constants; additive offsets do not affect stopping") {
    implicit val ss: SparkSession = spark
    for (sense <- Seq(Minimize, Maximize)) {
      def run(offset: Double): (Int, Double) = {
        val model = LpProblem("progress", sense)
        val x = model.variable("x", lowerBound = 5.0, upperBound = Some(10.0))
        model += 2.0 * x + offset
        model += (x <= 10.0).named("cap")
        val events = ArrayBuffer.empty[SolveProgress]

        val result = model.solve(SolveConfig(tolerance = 1e-30,
          control = SolveControl(onProgress = events += _, stagnation = Some(
            StagnationConfig(patience = 2, absoluteImprovement = 100.0, relativeImprovement = 0.0)))))
        try {
          assert(result.status == LpStatus.Stopped)
          assert(result.stopReason.contains(StopReason.NoProgress))
          assert(result.candidate.available)
          assert(result.candidate.feasible)
          assert(math.abs(result.objectiveValue - (2.0 * result.value(x) + offset)) < 1e-6)
          val retained = events.find(e => e.phase == SolvePhase.OuterIteration &&
            e.iteration == result.candidate.iteration.get).get

          assert(retained.iterate.map(_.objectiveValue).contains(result.objectiveValue))
          assert(retained.iterate.map(_.primalResidual).contains(result.residuals.primal))
          assert(retained.iterate.map(_.dualResidual).contains(result.residuals.dual))
          assert(retained.iterate.map(_.dualityGap).contains(result.residuals.gap))
          (result.iterations, result.value(x))
        } finally result.close()
      }
      val a = run(7.0)
      val b = run(1000000007.0)

      assert(a == b)
    }
  }

  test("stopping before initialization exposes no values, objective or invented activity") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("absent", Minimize)
    val x = model.variable("x")
    model += lpSum(x)
    model += (x >= 1.0).named("floor")

    val result = model.solve(SolveConfig(control = SolveControl(shouldStop = () => true)))
    try {
      assert(result.status == LpStatus.Stopped)
      assert(result.iterations == 0)
      assert(result.candidate == CandidateInfo.Unavailable)
      assert(result.objectiveValue.isNaN)
      assert(result.constraints.select("activity").head().getDouble(0).isNaN)
      assert(intercept[LpModelException](result.value(x)).getMessage.contains("No completed iterate"))
    } finally result.close()

    assert(spark.range(1).count() == 1)
  }

  test("candidate feasibility covers original bounds, free splits, fixed and merged rows") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("original-units", Minimize)
    val x = model.variable("x", lowerBound = -100.0, upperBound = Some(-90.0))
    val y = model.variable("y", lowerBound = Double.NegativeInfinity)
    val fixed = model.variable("fixed", lowerBound = 3.0, upperBound = Some(3.0))
    model += lpSum(x)
    model += (x + y + fixed === 0.0).named("balance")
    model += (2.0 * x + 2.0 * y + 2.0 * fixed === 0.0).named("duplicate")
    model += (fixed === 3.0).named("fixed")

    val solution = model.solve()
    try {
      assert(solution.candidate.feasible)
      val xv = solution.value(x)

      assert(xv >= -100.0 - 1e-6)
      assert(xv <= -90.0 + 1e-6)
      assert(math.abs(xv + solution.value(y) + 3.0) < 1e-8)
      assert(solution.constraints.count() == 3)
    } finally solution.close()
  }

  test("new controls are rejected for integer models, including fixed integer presolve") {
    implicit val ss: SparkSession = spark
    for (fixed <- Seq(false, true); control <- Seq(
      SolveControl(onProgress = _ => ()),
      SolveControl(shouldStop = () => false),
      SolveControl(timeLimit = Some(1.second)),
      SolveControl(stagnation = Some(StagnationConfig())))) {
      val model = LpProblem("integer-control", Minimize)
      val x = model.variable("x", lowerBound = if (fixed) 1.0 else 0.0,
        upperBound = Some(1.0), category = Integer)
      model += lpSum(x)
      model += (x >= 1.0).named("floor")

      assert(intercept[LpModelException](model.solve(SolveConfig(control = control)))
        .getMessage.contains("only for continuous"))
    }
  }

  test("fixed presolve respects a stricter candidate feasibility tolerance") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("fixed-tolerance", Minimize)
    val x = model.variable("x", lowerBound = 1.0, upperBound = Some(1.0))
    model += lpSum(x)
    model += (x === 1.0 + 1e-12).named("almost")

    val result = model.solve(SolveConfig(control = SolveControl(feasibilityTolerance = 1e-14)))
    try {
      assert(result.status == LpStatus.Optimal) // unchanged presolve contract
      assert(result.candidate.available)
      assert(!result.candidate.feasible)
      assert(result.value(x) == 1.0)
    } finally result.close()
  }
}
