package com.github.vbmacher.spark_lp

import TestingUtils._
import com.github.vbmacher.spark_lp.vectors.{DMatrix, DVector}
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.scalatest.funsuite.AnyFunSuite

class InitializeSuite extends AnyFunSuite with DataFrameSuiteBase {
  for (backend <- Seq("direct", "matrix-free CG")) {
    test(s"$backend initialization produces the reference primal, dual and slack vectors") {
      val costs = Array(2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0)
      val rows = Seq(
        Vectors.dense(12.0, 16.0, 30.0, 1.0, 0.0),
        Vectors.dense(24.0, 16.0, 12.0, 0.0, 1.0),
        Vectors.dense(-1.0, 0.0, 0.0, 0.0, 0.0),
        Vectors.dense(0.0, -1.0, 0.0, 0.0, 0.0),
        Vectors.dense(0.0, 0.0, -1.0, 0.0, 0.0),
        Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0),
        Vectors.dense(0.0, 0.0, 0.0, 0.0, 1.0))
      val c: DVector = sc.parallelize(costs, 2).glom.map(new DenseVector(_))
      val matrix: DMatrix = sc.parallelize(rows, 2)
      val rhs = new DenseVector(Array(120.0, 120.0, 120.0, 15.0, 15.0))
      // Independent dense least-squares reference, including positivity shifts.
      val expectedPrimal = Vectors.dense(
        35.602943519691436, 37.0752546522384, 40.48558919864948, 18.473136881378853,
        38.16880049342744, 45.485961891243974, 44.013650758696976)
      val expectedDual = Vectors.dense(
        0.02341386416450997, 0.026132327794100185, 0.04327618713552792,
        0.00131538562721981, 0.0003178848599113275)
      val expectedSlack = Vectors.dense(
        0.009879685443294282, 0.008882184675985859, 0.03197816398058437, 0.03469662761017458,
        0.051840486951602316, 0.007248914188854583, 0.008246414956163066)

      val result = if (backend == "direct") Initialize.init(c, matrix, rhs)
        else Initialize.init(c, matrix, rhs, new newton.CgFactory(1e-12, 0)(spark))

      assert(Vectors.dense(result.x.flatMap(_.toArray).collect()) ~= expectedPrimal relTol 1e-6)
      assert(Vectors.dense(result.lambda.toArray) ~= expectedDual relTol 1e-6)
      assert(Vectors.dense(result.s.flatMap(_.toArray).collect()) ~= expectedSlack relTol 1e-6)
    }
  }

  test("initialization shifts primal and slack vectors strictly above zero") {
    val costs: DVector = sc.parallelize(Array(0.0, 1.0), 1).glom.map(new DenseVector(_))
    val matrix: DMatrix = sc.parallelize(Seq(Vectors.dense(1.0), Vectors.dense(2.0)), 1)
    val rhs = new DenseVector(Array(-1.0))

    val result = Initialize.init(costs, matrix, rhs)

    assert(result.x.flatMap(_.values).collect().forall(_ > 0.0))
    assert(result.s.flatMap(_.values).collect().forall(_ > 0.0))
  }
}
