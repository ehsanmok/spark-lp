package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.SolveControl
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpPrioritiesSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("priorities preserve negative objectives including constants in both senses and repeated calls") {
    implicit val ss: SparkSession = spark
    for (sense <- Seq(Minimize, Maximize)) {
      val model = LpProblem("priorities", sense)
      val x = model.variable("x", upperBound = Some(10.0))
      val y = model.variable("y", upperBound = Some(10.0))
      model += (x + y === 10.0)
      model += lpSum(x)
      val first = if (sense == Minimize) x - 5.0 else x - 15.0
      for (_ <- 1 to 2) {
        val result = model.solvePriorities(Seq(LpPriority(first, 0.5, 0.1), LpPriority(lpSum(y))))
        try {
          assert(result.complete)
          assert(result.stages.size == 2)
          val finalStage = result.stages.last.solution.get
          val xv = finalStage.value(result.copied.variable(x))

          assert(math.abs(xv - (if (sense == Minimize) 1.0 else 9.0)) < 1e-5)
          assert(model.constraints.size == 1)
          assert(model.handles.size == 2)
        } finally result.close()
      }
    }
  }

  test("zero optimum, interrupted first stage and unbounded second stage retain explicit outcomes") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("partial")
    val x = model.variable("x")
    val y = model.variable("y")
    model += (x <= 2.0)
    val priorities = Seq(LpPriority(lpSum(x), absoluteTolerance = 1e-7), LpPriority(-1.0 * y))

    val result = model.solvePriorities(priorities)
    try {
      assert(!result.complete)
      assert(result.stages.size == 2)
      assert(result.stages.head.solution.get.status == LpStatus.Optimal)
      assert(Set[LpStatus](LpStatus.Unbounded, LpStatus.InfeasibleOrUnbounded)
        .contains(result.stages.last.solution.get.status))
    } finally result.close()

    val stopped = model.solvePriorities(priorities, SolveConfig(control = SolveControl(shouldStop = () => true)))
    try {
      assert(!stopped.complete)
      assert(stopped.stages.size == 1)
      assert(!stopped.stages.head.solution.get.candidate.available)
    } finally stopped.close()

    intercept[IllegalArgumentException](model.solvePriorities(Seq.empty))
    for (t <- Seq(-1.0, Double.NaN, Double.PositiveInfinity)) {
      intercept[IllegalArgumentException](LpPriority(lpSum(x), t))
      intercept[IllegalArgumentException](LpPriority(lpSum(x), relativeTolerance = t))
    }
  }
}
