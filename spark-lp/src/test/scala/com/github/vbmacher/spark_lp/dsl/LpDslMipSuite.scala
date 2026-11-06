package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.TestingUtils._
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

/** Integer/Binary variables solve through the same public API as continuous variables. */
class LpDslMipSuite extends AnyFunSuite with DataFrameSuiteBase {

  test("Integer and Binary variables use the standard solve path") {
    implicit val ss: SparkSession = spark
    for (category <- Seq(Integer, Binary)) {
      val model = LpProblem("cat", Minimize)
      val x = model.variable("x", upperBound = Some(1.0), category = category)
      model += lpSum(x)
      model += (x >= 1.0).named("floor")

      val solution = model.solve()

      assert(solution.status == LpStatus.Optimal)
      assert(solution.value(x) == 1.0)
    }
  }

  test("bounded Integer variables: max x + y with 2x + 2y <= 3 is 1, not the relaxation's 1.5") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("boundedInt", Maximize)
    val x = model.variable("x", upperBound = Some(5.0), category = Integer)
    val y = model.variable("y", upperBound = Some(5.0), category = Integer)
    model += x + y
    model += (2.0 * x + 2.0 * y <= 3.0).named("cap")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 1.0 absTol 1e-6)
    val (vx, vy) = (solution.value(x), solution.value(y))

    assert(vx == math.rint(vx) && vy == math.rint(vy), s"values must be integral, got ($vx, $vy)")
    assert(vx + vy == 1.0)
  }

  test("binary knapsack over a DataFrame domain picks the optimal subset") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val items = Seq(("a", 10.0, 6.0), ("b", 7.0, 5.0), ("c", 6.0, 5.0)).toDF("item", "value", "weight")

    val model = LpProblem("knapsack", Maximize)
    val pick = model.variables("pick", items, $"item", category = Binary)
    model += lpSum(pick * $"value")
    model += (lpSum(pick * $"weight") <= 10.0).named("capacity")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    // LP relaxation packs 'a' plus a fraction of 'b' (15.6); the integer optimum is {b, c} = 13

    assert(solution.objectiveValue ~== 13.0 absTol 1e-6)

    val values = solution.values(pick)

    assert(values.columns.toSeq == Seq("item", "value", "weight", "lp_variable", "lp_value"))
    val byItem = values.select("item", "lp_value").collect()
      .map(r => r.getString(0) -> r.getDouble(1)).toMap

    assert(byItem == Map("a" -> 0.0, "b" -> 1.0, "c" -> 1.0))
  }

  test("mixed model: Integer x and continuous y keep integrality only where declared") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("mixed", Maximize)
    val x = model.variable("x", upperBound = Some(10.0), category = Integer)
    val y = model.variable("y", upperBound = Some(4.5))
    model += 2.0 * x + y
    model += (x + y <= 7.3).named("budget")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 14.3 absTol 1e-5)
    assert(solution.value(x) == 7.0, "the Integer variable must be exactly integral")
    assert(solution.value(y) ~== 0.3 absTol 1e-5)
  }

  test("integer-infeasible model is truthfully Infeasible: 10 <= 3x <= 11 with Integer x") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("intInfeasible", Minimize)
    val x = model.variable("x", upperBound = Some(10.0), category = Integer)
    model += lpSum(x)
    model += (3.0 * x >= 10.0).named("lo")
    model += (3.0 * x <= 11.0).named("hi")

    // the relaxation is feasible (10/3 <= x <= 11/3) but holds no integer;
    // both branches (x <= 3, x >= 4) carry Farkas certificates

    val solution = model.solve()

    assert(solution.status == LpStatus.Infeasible)
    assert(solution.objectiveValue.isNaN)
  }

  test("Binary bounds intersect {0, 1}: lowerBound = 1 pins the variable to exactly 1") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("pinnedBinary", Minimize)
    val x = model.variable("x", lowerBound = 1.0, category = Binary)
    val y = model.variable("y", upperBound = Some(10.0))
    model += x + y
    model += (y - x >= 0.0).named("chain")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 2.0 absTol 1e-6)
    assert(solution.value(x) == 1.0)
    assert(solution.value(y) ~== 1.0 absTol 1e-5)
  }

  test("a separable Integer objective needs no finite upper bound") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("noUb", Minimize)
    val x = model.variable("x", category = Integer)
    model += lpSum(x)
    model += (x >= 1.0).named("floor")

    val solution = model.solve()
    try {
      assert(solution.status == LpStatus.Optimal)
      assert(solution.value(x) == 1.0)
    }
    finally solution.close()
  }

  test("Binary whose declared bounds exclude {0, 1} entirely is rejected") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("emptyBinary", Minimize)
    val x = model.variable("x", lowerBound = 2.0, category = Binary)
    model += lpSum(x)
    model += (x >= 0.0).named("floor")

    val error = intercept[LpModelException](model.solve())

    assert(error.getMessage.contains("no integral values"))
  }
}
