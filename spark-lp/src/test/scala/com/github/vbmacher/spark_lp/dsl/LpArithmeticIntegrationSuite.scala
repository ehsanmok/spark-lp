package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpArithmeticIntegrationSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("division and variable dot products solve the manually expanded model") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("arithmetic")
    val x = model.variable("x")
    val y = model.variable("y")
    val variables = Seq(x, y)
    model += lpDot(Seq(2.0, 3.0), variables) / 2.0 + 4.0
    model += (x / -2.0 <= -1.0)
    model += (y >= 1.0)

    val result = model.solve()

    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.value(x) - 2.0) < 1e-6)
      assert(math.abs(result.value(y) - 1.0) < 1e-6)
      assert(math.abs(result.objectiveValue - 7.5) < 1e-6)
    } finally result.close()
  }
}
