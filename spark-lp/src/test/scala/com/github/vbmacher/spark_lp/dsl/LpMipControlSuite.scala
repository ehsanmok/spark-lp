package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.StopReason
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import scala.concurrent.duration._

class LpMipControlSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("injected deadlines before a node and inside a node expose no fractional incumbent") {
    implicit val ss: SparkSession = spark
    for (inside <- Seq(false, true)) {
      val model = LpProblem("control", Maximize)
      val x = model.variable("x", upperBound = Some(10.0), category = Integer)
      model += x + 7.0
      model += (3.0 * x <= 10.0)
      var clock = 0L
      val control = MipControl(timeLimit = Some(1.second), nanoTime = () => clock,
        onProgress = _ => if (!inside) clock = 2000000000L,
        onNodeProgress = (_, _) => if (inside) clock = 2000000000L)

      val result = model.solve(SolveConfig(mip = MipConfig(control = control)))
      try {
        assert(result.status == LpStatus.Stopped)
        assert(result.stopReason.contains(StopReason.TimeLimit))
        assert(!result.candidate.available)
        assert(result.objectiveValue.isNaN)
        assert(result.mip.get.incumbent.isEmpty)
        assert(result.mip.get.absoluteGap.isEmpty)
        intercept[LpModelException](result.value(x))
      } finally result.close()
    }
  }

  test("cancellation after an incumbent preserves matching exact integer values and objective") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("control", Maximize)
    val x = model.variable("x", upperBound = Some(10.0), category = Integer)
    model += x + 7.0
    model += (3.0 * x <= 10.0)
    var cancel = false
    var sawIncumbent = false

    val result = model.solve(SolveConfig(mip = MipConfig(control = MipControl(
      shouldStop = () => cancel,
      onProgress = event => if (event.incumbent.nonEmpty && event.openNodes > 0) {
        sawIncumbent = true; cancel = true
      }))))
    try {
      assert(sawIncumbent)
      assert(result.status == LpStatus.Stopped)
      assert(result.stopReason.contains(StopReason.UserRequested))
      assert(result.candidate.available)
      assert(result.candidate.feasible)
      assert(result.value(x) == 3.0)
      assert(result.objectiveValue == 10.0)
      assert(result.mip.get.incumbent.contains(10.0))
    } finally result.close()
  }

  test("deadline after incumbent and callback exceptions preserve resource ownership") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("control", Maximize)
    val x = model.variable("x", upperBound = Some(10.0), category = Integer)
    model += x + 7.0
    model += (3.0 * x <= 10.0)
    var clock = 0L

    val result = model.solve(SolveConfig(mip = MipConfig(control = MipControl(
      timeLimit = Some(1.second), nanoTime = () => clock,
      onProgress = event => if (event.incumbent.nonEmpty && event.openNodes > 0) clock = 2000000000L))))
    try {
      assert(result.status == LpStatus.Stopped)
      assert(result.stopReason.contains(StopReason.TimeLimit))
      assert(result.value(x) == 3.0)
      assert(result.candidate.feasible)
    } finally result.close()
    val owned = spark.range(10).cache()
    owned.count()
    val failure = new IllegalStateException("callback failure")

    val thrown = intercept[IllegalStateException](model.solve(SolveConfig(mip = MipConfig(
      control = MipControl(onNodeProgress = (_, _) => throw failure)))))

    assert(thrown eq failure)
    assert(owned.count() == 10)
    assert(owned.storageLevel.useMemory)
    owned.unpersist()
  }
}
