package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpReducedCostSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("reduced costs satisfy original bound KKT signs in both objective senses") {
    implicit val ss: SparkSession = spark
    for (sense <- Seq(Minimize, Maximize)) {
      val model = LpProblem("reduced", sense)
      val lower = model.variable("lower", 1.0, Some(4.0))
      val upper = model.variable("upper", -1.0, Some(3.0))
      val free = model.variable("free", Double.NegativeInfinity)
      val fixed = model.variable("fixed", 2.0, Some(2.0))
      val sign = if (sense == Minimize) 1.0 else -1.0
      model += sign * (2.0 * lower - 3.0 * upper + 4.0 * free + fixed)
      model += (free === -2.0)

      val result = model.solve()
      try {
        assert(result.status == LpStatus.Optimal)
        assert(math.abs(result.reducedCost(lower).get - sign * 2.0) < 1e-6)
        assert(math.abs(result.reducedCost(upper).get + sign * 3.0) < 1e-6)
        assert(math.abs(result.reducedCost(free).get) < 1e-6)
        assert(result.reducedCost(fixed).isEmpty)
      } finally result.close()
    }
  }

  test("keyed sensitivity is distributed, stopped results are unavailable and closure is enforced") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("keyed")
    val xs = model.variables("x", Seq("a", "b").toDF("key"), $"key", upperBound = Some(3.0))
    model += xs.sum
    model += (xs("a") >= 1.0)

    val result = model.solve()
    try {
      assert(math.abs(result.reducedCost(xs("a")).get) < 1e-6)
      assert(math.abs(result.reducedCost(xs("b")).get - 1.0) < 1e-6)
      assert(result.reducedCosts(xs).count() == 2)
      result.close()

      intercept[LpModelException](result.reducedCost(xs("a")))
    } finally result.close()

    val stopped = model.solve(SolveConfig(stopAfterIteration = Some(_ => true)))
    try {
      assert(stopped.reducedCost(xs("a")).isEmpty)
      assert(stopped.reducedCosts(xs).filter("lp_reduced_cost IS NOT NULL").count() == 0)
    } finally stopped.close()
  }
}
