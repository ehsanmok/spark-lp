package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.scalatest.funsuite.AnyFunSuite

class LpUpperOnlySuite extends AnyFunSuite with DataFrameSuiteBase {
  test("upper-only scalar bounds preserve signs and constants in both objective senses") {
    implicit val ss: SparkSession = spark
    for (upper <- Seq(-3.0, 0.0, 5.0); sense <- Seq(Minimize, Maximize)) {
      val sign = if (sense == Minimize) -1.0 else 1.0
      val model = LpProblem("upper", sense)
      val x = model.variable("x", lowerBound = Double.NegativeInfinity, upperBound = Some(upper))
      model += sign * x + 7.0

      val result = model.solve()
      try {
        assert(result.status == LpStatus.Optimal)
        assert(result.value(x) == upper)
        assert(result.objectiveValue == sign * upper + 7.0)
      } finally result.close()
    }
  }

  test("keyed reflection agrees with explicit free-variable constraints") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    def solve(reflect: Boolean): Double = {
      val model = LpProblem("keyed", Maximize)
      val xs = model.variables("xs", Seq("a", "b").toDF("id"), col("id"),
        lowerBound = Double.NegativeInfinity, upperBound = if (reflect) Some(3.0) else None)
      model += xs.sum + 4.0
      model += (xs.sum >= 0.0)
      if (!reflect) model += (xs.sumBy("id")() <= 3.0)

      val result = model.solve(SolveConfig(newtonSolver = NewtonSolver.ConjugateGradient))
      try {
        assert(result.status == LpStatus.Optimal)
        val values = result.values(xs).select("lp_value").collect().map(_.getDouble(0))

        assert(values.forall(v => math.abs(v - 3.0) < 1e-6))
        result.objectiveValue
      } finally result.close()
    }

    assert(math.abs(solve(true) - solve(false)) < 1e-6)
  }

  test("reflection preserves quadratic constants and infeasibility evidence") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("quadratic")
    val x = model.variable("x", lowerBound = Double.NegativeInfinity, upperBound = Some(-1.0))
    model += QpObjective.squaredDeviation(x, -2.0) + LpExpr.constant(3.0)

    val result = model.solve()
    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.value(x) + 2.0) < 1e-6)
      assert(math.abs(result.objectiveValue - 3.0) < 1e-6)
    } finally result.close()
    model += (x >= 0.0)

    val infeasible = model.solve()
    try {
      assert(infeasible.status == LpStatus.Infeasible)
      assert(infeasible.evidence.exists(_.verify(1e-7).valid))
    } finally infeasible.close()
  }

  test("an improving direction below the upper bound is unbounded") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("unbounded")
    val x = model.variable("x", lowerBound = Double.NegativeInfinity, upperBound = Some(2.0))
    model += x + 4.0

    val result = model.solve()
    try {
      assert(result.status == LpStatus.Unbounded)
      assert(result.value(x) <= 2.0)
    } finally result.close()
  }
}
