package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.{SolveControl, StopReason}
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpSolveTimingsSuite extends AnyFunSuite with DataFrameSuiteBase {
  private final class TestClock(wallValues: Seq[Long], cpuValues: Seq[Option[Long]]) {
    private val walls = wallValues.iterator
    private val cpus = cpuValues.iterator
    var wallCalls = 0
    var cpuCalls = 0
    val clock = new LpSolveClock(
      () => { wallCalls += 1; walls.next() },
      () => { cpuCalls += 1; cpus.next() })
  }

  test("presolve-only results report compilation, reconstruction and driver CPU time") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("fixed timing", Minimize)
    val x = model.variable("x", lowerBound = 2.0, upperBound = Some(2.0))
    model += x
    val source = new TestClock(
      Seq(0L, 2000000000L, 5000000000L),
      Seq(Some(10000000000L), Some(14000000000L)))

    val result = model.solve(SolveConfig(), source.clock)
    try {
      assert(result.status == LpStatus.Optimal)
      assert(result.timings == LpSolveTimings(2.0, 0.0, 3.0, 5.0, Some(4.0)))
      assert(source.wallCalls == 3)
      assert(source.cpuCalls == 2)
    } finally result.close()
  }

  test("stopped continuous results retain deterministic phase timings when CPU time is unavailable") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("stopped timing", Minimize)
    val x = model.variable("x", upperBound = Some(10.0))
    model += x
    model += (x >= 1.0)
    val source = new TestClock(
      Seq(0L, 1000000000L, 4000000000L, 10000000000L),
      Seq(None, None))

    val result = model.solve(SolveConfig(control = SolveControl(shouldStop = () => true)), source.clock)
    try {
      assert(result.status == LpStatus.Stopped)
      assert(result.stopReason.contains(StopReason.UserRequested))
      assert(result.timings == LpSolveTimings(1.0, 3.0, 6.0, 10.0, None))
      assert(source.wallCalls == 4)
      assert(source.cpuCalls == 2)
    } finally result.close()
  }

  test("MIP timing uses the outer solve interval instead of accumulating search durations") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("MIP timing", Maximize)
    val x = model.variable("x", upperBound = Some(10.0), category = Integer)
    model += x + 7.0
    model += (3.0 * x <= 10.0)
    var searchNanos = 0L
    val searchClock = () => { searchNanos += 1000000000000L; searchNanos }
    val source = new TestClock(
      Seq(0L, 2000000000L, 7000000000L, 11000000000L),
      Seq(Some(20000000000L), Some(29000000000L)))

    val config = SolveConfig(mip = MipConfig(control = MipControl(
      shouldStop = () => true, nanoTime = searchClock)))

    val result = model.solve(config, source.clock)
    try {
      assert(result.status == LpStatus.Stopped)
      assert(result.mip.exists(m => m.processedNodes == 0 && m.elapsedSeconds >= 1000.0))
      assert(result.timings == LpSolveTimings(2.0, 5.0, 4.0, 11.0, Some(9.0)))
      assert(source.wallCalls == 4)
      assert(source.cpuCalls == 2)
    } finally result.close()
  }
}
