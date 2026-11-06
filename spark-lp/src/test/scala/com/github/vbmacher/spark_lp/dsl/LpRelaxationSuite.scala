package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.SolveControl
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpRelaxationSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("relaxation exposes an integrality gap and ordinary subsequent solves retain categories") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("gap", Maximize)
    val x = model.variable("x", category = Binary)
    val y = model.variable("y", category = Binary)
    model += x + y + 3.0
    model += (2.0 * x + 2.0 * y <= 3.0)

    val relaxed = model.solve(SolveConfig(relaxIntegrality = true))
    val integral = model.solve()
    try {
      assert(relaxed.isRelaxation)
      assert(!integral.isRelaxation)
      assert(relaxed.status == LpStatus.Optimal)
      assert(integral.status == LpStatus.Optimal)
      assert(math.abs(relaxed.objectiveValue - 4.5) < 1e-6)
      assert(integral.objectiveValue == 4.0)
      assert(math.abs(relaxed.value(x) + relaxed.value(y) - 1.5) < 1e-6)
      assert(x.handle.category == Binary)
      assert(x.handle.upperBound.isEmpty)
    } finally { relaxed.close(); integral.close() }
  }

  test("effective integer and binary bounds tighten before relaxation while mixed continuous bounds remain") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("bounds")
    val x = model.variable("x", lowerBound = 0.2, upperBound = Some(3.8), category = Integer)
    val b = model.variable("b", lowerBound = 0.2, category = Binary)
    val y = model.variable("y", lowerBound = 0.2)
    model += x + b + y
    model += (x + y >= 1.0)

    val result = model.solve(SolveConfig(relaxIntegrality = true))
    try {
      assert(result.isRelaxation)
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.objectiveValue - 2.2) < 1e-6)
      assert(math.abs(result.value(x) - 1.0) < 1e-6)
      assert(result.value(b) == 1.0)
    } finally result.close()

    val stopped = model.solve(SolveConfig(relaxIntegrality = true,
      control = SolveControl(shouldStop = () => true)))
    try {
      assert(stopped.isRelaxation)
      assert(!stopped.candidate.available)
    } finally stopped.close()
  }
}
