package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpDualSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("LP shadow prices match objective derivatives in both senses and row directions") {
    implicit val ss: SparkSession = spark
    for (sense <- Seq(Minimize, Maximize); equality <- Seq(false, true)) {
      def solve(rhs: Double): LpSolution = {
        val model = LpProblem("dual", sense)
        val x = model.variable("x", lowerBound = -2.0, upperBound = Some(10.0))
        model += 3.0 * x + 7.0
        model += (if (equality) x === rhs else if (sense == Minimize) x >= rhs else x <= rhs)
        model.solve()
      }
      val base = solve(2.0)
      val shifted = solve(2.001)
      try {
        val price = base.constraints.first().getAs[Double]("dual")

        assert(math.abs(price - 3.0) < 1e-5)
        assert(math.abs((shifted.objectiveValue - base.objectiveValue) / 0.001 - price) < 1e-4)
        assert(base.constraints.first().isNullAt(base.constraints.schema.fieldIndex("dual_note")))
      } finally { base.close(); shifted.close() }
    }
  }

  test("removed and merged rows, stopped solves and MIPs explicitly lack global shadow prices") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("merged")
    val x = model.variable("x")
    val y = model.variable("fixed", 1.0, Some(1.0))
    model += lpSum(x)
    model += (x === 2.0).named("a")
    model += (2.0 * x === 4.0).named("b")
    model += (y === 1.0).named("fixed")

    val result = model.solve()
    try {
      assert(result.constraints.filter("dual IS NOT NULL").count() == 0)
      assert(result.constraints.filter("dual_note IS NULL").count() == 0)
    } finally result.close()
    val mip = LpProblem("integer")
    val z = mip.variable("z", upperBound = Some(3.0), category = Integer)
    mip += lpSum(z)
    mip += (z >= 1.0)

    val integer = mip.solve()
    try {
      assert(integer.constraints.filter("dual IS NOT NULL").count() == 0)
    } finally integer.close()

    val stopped = model.solve(SolveConfig(stopAfterIteration = Some(_ => true)))
    try {
      assert(stopped.constraints.filter("dual IS NOT NULL").count() == 0)
    } finally stopped.close()
  }
}
