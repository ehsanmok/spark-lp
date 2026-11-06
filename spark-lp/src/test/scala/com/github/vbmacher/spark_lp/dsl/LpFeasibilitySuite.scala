package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.SolveControl
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpFeasibilitySuite extends AnyFunSuite with DataFrameSuiteBase {
  test("constraint-only models find feasible continuous, integer and binary assignments") {
    implicit val ss: SparkSession = spark
    for (category <- Seq(Continuous, Integer, Binary); sense <- Seq(Minimize, Maximize)) {
      val model = LpProblem("feasibility", sense)
      val x = model.variable("x", upperBound = Some(3.0), category = category)
      model += (x >= 0.75)

      val solution = model.solve()
      try {
        assert(solution.status == LpStatus.Optimal)
        assert(solution.objectiveValue == 0.0)
        assert(solution.candidate.feasible)
        assert(solution.value(x) >= 0.75 - 1e-8)
        if (category != Continuous) assert(solution.value(x) == math.rint(solution.value(x)))
      } finally solution.close()

      assert(model.objective.isEmpty)
    }
  }

  test("contradictory constraints have a conclusive outcome with zero objective") {
    implicit val ss: SparkSession = spark
    for (category <- Seq(Continuous, Integer, Binary)) {
      val model = LpProblem("infeasible")
      val x = model.variable("x", upperBound = Some(3.0), category = category)
      model += (x >= 0.75)
      model += (x <= 0.25)

      val solution = model.solve()
      try {
        assert(solution.status == LpStatus.Infeasible)
      } finally solution.close()
    }
  }

  test("unbounded feasible regions have bounded zero objective and allow subsequent objectives") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("region")
    val x = model.variable("x")
    model += (x >= 2.0)

    val feasible = model.solve()
    try {
      assert(feasible.status == LpStatus.Optimal)
      assert(feasible.objectiveValue == 0.0)
    } finally feasible.close()
    model += x + 3.0

    val optimized = model.solve()
    try {
      assert(math.abs(optimized.objectiveValue - 5.0) < 1e-6)
    } finally optimized.close()
  }

  test("early termination preserves explicit unavailable-candidate semantics") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("stopped")
    val x = model.variable("x")
    model += (x >= 2.0)

    val result = model.solve(SolveConfig(control = SolveControl(shouldStop = () => true)))
    try {
      assert(result.status == LpStatus.Stopped)
      assert(!result.candidate.available)
      intercept[LpModelException](result.value(x))
    } finally result.close()
  }
}
