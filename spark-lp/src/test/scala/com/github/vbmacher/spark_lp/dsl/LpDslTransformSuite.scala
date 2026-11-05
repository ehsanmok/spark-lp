package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.TestingUtils._
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

/** Equality-form transformations: bound shifts, objective constants, fixed presolve, free split. */
class LpDslTransformSuite extends AnyFunSuite with DataFrameSuiteBase {

  test("lower-bound shift restores the objective constant: minimize x with 5 <= x <= 10 is 5") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("shift", Minimize)
    val x = model.variable("x", lowerBound = 5.0, upperBound = Some(10.0))
    model += lpSum(x)
    model += (x <= 10.0).named("cap")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 5.0 absTol 1e-6)
    assert(solution.value(x) ~== 5.0 absTol 1e-6)
  }

  test("explicit objective constants survive for Minimize: 2x + 7 with x in [5, 10] is 17") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("constMin", Minimize)
    val x = model.variable("x", lowerBound = 5.0, upperBound = Some(10.0))
    model += 2.0 * x + 7.0
    model += (x <= 10.0).named("cap")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 17.0 absTol 1e-6)
    assert(solution.value(x) ~== 5.0 absTol 1e-6)
  }

  test("explicit objective constants survive for Maximize: 2x + 7 with x in [5, 10] is 27") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("constMax", Maximize)
    val x = model.variable("x", lowerBound = 5.0, upperBound = Some(10.0))
    model += 2.0 * x + 7.0
    model += (x <= 10.0).named("cap")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 27.0 absTol 1e-6)
    assert(solution.value(x) ~== 10.0 absTol 1e-6)
  }

  test("finite upper bound is honoured as an internal bound row, not a diagnostics row") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val model = LpProblem("ub", Maximize)
    val domain = Seq("a", "b").toDF("k")
    val v = model.variables("v", domain, $"k", upperBound = Some(5.0))
    model += lpSum(v)
    model += (lpSum(v) <= 100.0).named("loose")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 10.0 absTol 1e-5)
    val values = solution.values(v).select("k", "lp_value").collect()
      .map(r => r.getString(0) -> r.getDouble(1)).toMap

    assert(values("a") ~== 5.0 absTol 1e-5)
    assert(values("b") ~== 5.0 absTol 1e-5)
    // bound rows are internal: only the user constraint appears in diagnostics

    assert(solution.constraints.collect().map(_.getString(0)).toSeq == Seq("loose"))
  }

  test("lowerBound == upperBound is presolved: exact value, no emitted row for fixed-only constraints") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("fixed", Minimize)
    val f = model.variable("f", lowerBound = 3.0, upperBound = Some(3.0))
    val y = model.variable("y")
    val z = model.variable("z")
    model += y + 2.0 * z
    model += (f + y + z === 8.0).named("uses_fixed")
    model += (lpSum(f) === 3.0).named("fixed_only")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.value(f) == 3.0, "a fixed variable reports its bound exactly, not approximately")
    assert(solution.value(y) ~== 5.0 absTol 1e-4)
    assert(solution.value(z) ~== 0.0 absTol 1e-4)
    assert(solution.objectiveValue ~== 5.0 absTol 1e-4)

    val rows = solution.constraints.collect().map(r => r.getString(0) -> r).toMap
    // the fixed-only row was never sent to the solver, but stays visible with a note
    val fixedOnly = rows("fixed_only")

    assert(!fixedOnly.isNullAt(7))
    assert(fixedOnly.getString(7).contains("presolved"))
    assert(fixedOnly.getDouble(2) ~== 3.0 absTol 1e-12, "activity includes the fixed contribution")
    // the mixed row reports full activity including the fixed variable

    assert(rows("uses_fixed").getDouble(2) ~== 8.0 absTol 1e-4)
  }

  test("fixed variables contribute c*l to the objective constant") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("fixedObj", Minimize)
    val f = model.variable("f", lowerBound = 4.0, upperBound = Some(4.0))
    val y = model.variable("y")
    model += 10.0 * f + y
    model += (y >= 2.0).named("floor")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 42.0 absTol 1e-5)
  }

  test("free variables split into two non-negative parts and reconstruct a negative value") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("free", Minimize)
    val x = model.variable("x", lowerBound = Double.NegativeInfinity)
    val y = model.variable("y")
    model += x + 2.0 * y
    model += (x + y === 3.0).named("sum")
    model += (x - y === -7.0).named("diff")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.value(x) ~== -2.0 absTol 1e-4)
    assert(solution.value(y) ~== 5.0 absTol 1e-4)
    assert(solution.objectiveValue ~== 8.0 absTol 1e-4)
  }

  test("an upper-only variable reconstructs a constrained value") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("freeUb", Minimize)
    val x = model.variable("x", lowerBound = Double.NegativeInfinity, upperBound = Some(1.0))
    model += lpSum(x)
    model += (x === 0.0).named("pin")

    val result = model.solve()
    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.value(x)) < 1e-8)
    } finally result.close()
  }

  test("lowerBound > upperBound is rejected") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("badBounds", Minimize)
    val x = model.variable("x", lowerBound = 2.0, upperBound = Some(1.0))
    model += lpSum(x)
    model += (x === 0.0).named("pin")

    val error = intercept[LpModelException](model.solve())

    assert(error.getMessage.contains("lowerBound"))
  }

  test("NaN and +inf lower bounds and non-finite upper bounds are rejected") {
    implicit val ss: SparkSession = spark

    def failing(lb: Double, ub: Option[Double]): LpModelException = {
      val model = LpProblem("nanBounds", Minimize)
      val x = model.variable("x", lowerBound = lb, upperBound = ub)
      model += lpSum(x)
      model += (x === 0.0).named("pin")

      intercept[LpModelException](model.solve())
    }

    assert(failing(Double.NaN, None).getMessage.contains("lower bound"))
    assert(failing(Double.PositiveInfinity, None).getMessage.contains("lower bound"))
    assert(failing(0.0, Some(Double.NaN)).getMessage.contains("upper bound"))
    assert(failing(0.0, Some(Double.PositiveInfinity)).getMessage.contains("upper bound"))
  }
}
