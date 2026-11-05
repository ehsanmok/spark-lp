package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.SolveControl
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.struct
import org.scalatest.funsuite.AnyFunSuite

class LpEvaluationSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("post-solve expressions combine scalar, repeated, composite-key and relational coefficients") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("evaluation")
    val x = model.variable("scalar", lowerBound = 2.0, upperBound = Some(2.0))
    val family = model.variables("family", Seq(("a", 1, 3.0), ("b", 2, 4.0)).toDF("g", "i", "c"), struct($"g", $"i"))
    model += family.sum
    model += (family(("a", 1)) >= 1.0)
    model += (family(("b", 2)) >= 2.0)

    val result = model.solve()
    try {
      val report = 2.0 * x + x + family.sum($"c") + family(("a", 1)) + 5.0

      assert(math.abs(result.evaluate(report) - 23.0) < 1e-6)
      assert(result.evaluate(lpSum(Seq.empty[LpExpr]).withConstant(7.0)) == 7.0)
      intercept[LpModelException](result.evaluate(LpProblem("foreign").variable("z") + 1.0))
      // A result can evaluate only variables present when it was solved.
      val later = model.variable("later")

      intercept[LpModelException](result.evaluate(lpSum(later)))
      // Closing a result also ends expression evaluation.
      result.close()

      intercept[LpModelException](result.evaluate(report))
    } finally result.close()
  }

  test("available stopped iterates may be evaluated without a feasibility claim; absent iterates fail") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("stopped")
    val x = model.variable("x")
    model += lpSum(x)
    model += (x >= 2.0)

    val stopped = model.solve(SolveConfig(stopAfterIteration = Some(_ => true)))
    try {
      assert(stopped.candidate.available)
      assert(math.abs(stopped.evaluate(3.0 * x + 1.0) - (3.0 * stopped.value(x) + 1.0)) < 1e-8)
    } finally stopped.close()

    val absent = model.solve(SolveConfig(control = SolveControl(shouldStop = () => true)))
    try intercept[LpModelException](absent.evaluate(lpSum(x))) finally absent.close()
  }
}
