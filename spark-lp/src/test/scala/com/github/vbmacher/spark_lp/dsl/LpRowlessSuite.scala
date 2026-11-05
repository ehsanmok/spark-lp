package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpRowlessSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("rowless lower-bounded and free variables have analytic optima or improving directions") {
    implicit val ss: SparkSession = spark
    for (sense <- Seq(Minimize, Maximize); lower <- Seq(0.0, Double.NegativeInfinity); cost <- Seq(-2.0, 0.0, 2.0)) {
      val model = LpProblem("rowless", sense)
      val x = model.variable("x", lowerBound = lower)
      model += cost * x + 3.0

      val result = model.solve()
      try {
        val sign = if (sense == Minimize) 1.0 else -1.0
        val unbounded = cost != 0 && (lower.isNegInfinity || sign * cost < 0)

        assert(result.status == (if (unbounded) LpStatus.Unbounded else LpStatus.Optimal))
        assert(result.objectiveValue == (if (unbounded) -sign * Double.PositiveInfinity else 3.0))
        assert(result.value(x) == 0.0)
        assert(result.iterations == 0)
        assert(result.candidate.available)
        assert(result.candidate.feasible)
      } finally result.close()
    }
  }

  test("finite, fixed and integral domains preserve exact endpoints and constants") {
    implicit val ss: SparkSession = spark
    for (category <- Seq(Continuous, Integer, Binary); sense <- Seq(Minimize, Maximize)) {
      val model = LpProblem("bounded", sense)
      val x = model.variable("x", lowerBound = 0.2, upperBound = Some(2.8), category = category)
      val fixed = model.variable("fixed", lowerBound = -3.0, upperBound = Some(-3.0))
      model += 2.0 * x + fixed + 7.0

      val result = model.solve()
      try {
        val expected = category match {
          case Continuous => if (sense == Minimize) 0.2 else 2.8
          case Integer => if (sense == Minimize) 1.0 else 2.0
          case Binary => 1.0
        }

        assert(result.status == LpStatus.Optimal)
        assert(result.iterations == 0)
        assert(result.value(x) == expected)
        assert(result.value(fixed) == -3.0)
        assert(result.objectiveValue == 2.0 * expected + 4.0)
      } finally result.close()
    }
  }

  test("empty models, zero-sized families and presolved rows return ordinary diagnostics") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val empty = LpProblem("constant")
    empty += LpExpr.constant(7.0)

    val constant = empty.solve()
    try {
      assert(constant.objectiveValue == 7.0)
      assert(constant.iterations == 0)
    } finally constant.close()
    val model = LpProblem("empty family")
    val xs = model.variables("xs", Seq.empty[String].toDF("id"), org.apache.spark.sql.functions.col("id"))
    model += xs.sum + 4.0
    model += (xs.sum === 0.0).named("empty equality")

    val result = model.solve()
    try {
      assert(result.objectiveValue == 4.0)
      assert(result.status == LpStatus.Optimal)
      assert(result.values(xs).count() == 0L)
      assert(result.constraints.select("activity").first().getDouble(0) == 0.0)
    } finally result.close()

    assert(spark.range(3).count() == 3)
  }
}
