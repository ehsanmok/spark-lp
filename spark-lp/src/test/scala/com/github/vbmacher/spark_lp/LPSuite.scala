package com.github.vbmacher.spark_lp

import com.github.vbmacher.spark_lp.newton.NewtonSolver

import TestingUtils._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.wrappers.CholeskyDecomposition
import org.scalatest.funsuite.AnyFunSuite

class LPSuite extends AnyFunSuite with DataFrameSuiteBase {

  private val numPartitions = 2

  test("direct LP solving recovers the reference allocation") {
    implicit val s: SparkSession = spark
    val cArray: Array[Double] = Array(2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    val BArray: Array[Array[Double]] = Array(
      Array(12.0, 16.0, 30.0, 1.0, 0.0),
      Array(24.0, 16.0, 12.0, 0.0, 1.0),
      Array(-1.0, 0.0, 0.0, 0.0, 0.0),
      Array(0.0, -1.0, 0.0, 0.0, 0.0),
      Array(0.0, 0.0, -1.0, 0.0, 0.0),
      Array(0.0, 0.0, 0.0, 1.0, 0.0),
      Array(0.0, 0.0, 0.0, 0.0, 1.0))
    val bArray: Array[Double] = Array(120.0, 120.0, 120.0, 15.0, 15.0)

    val c: RDD[DenseVector] = sc.parallelize(cArray, numPartitions).glom.map(new DenseVector(_))
    val rows: RDD[org.apache.spark.mllib.linalg.Vector] = sc.parallelize(BArray, numPartitions).map(Vectors.dense)
    val b = new DenseVector(bArray)

    val (_, allocation) = LP.solve(c, rows, b)
    // solution obtained from scipy.optimize.linprog and octave glgk lpsolver with fun_val = 12.083
    val expectedSol = Vectors.dense(
      Array(1.66666667, 5.83333333, 40.0, 0.0, 0.0, 13.33333333, 9.16666667))
    val actualAllocation = Vectors.dense(allocation.flatMap(_.toArray).collect())

    assert(actualAllocation ~== expectedSol absTol 1e-6, "LP.solve x should return the correct answer.")
  }

  test("LP solve with the matrix-free conjugate-gradient solver matches the direct solver") {
    implicit val s: SparkSession = spark
    val cArray: Array[Double] = Array(2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    val BArray: Array[Array[Double]] = Array(
      Array(12.0, 16.0, 30.0, 1.0, 0.0),
      Array(24.0, 16.0, 12.0, 0.0, 1.0),
      Array(-1.0, 0.0, 0.0, 0.0, 0.0),
      Array(0.0, -1.0, 0.0, 0.0, 0.0),
      Array(0.0, 0.0, -1.0, 0.0, 0.0),
      Array(0.0, 0.0, 0.0, 1.0, 0.0),
      Array(0.0, 0.0, 0.0, 0.0, 1.0))
    val bArray: Array[Double] = Array(120.0, 120.0, 120.0, 15.0, 15.0)

    val c: RDD[DenseVector] = sc.parallelize(cArray, numPartitions).glom.map(new DenseVector(_))
    val rows: RDD[org.apache.spark.mllib.linalg.Vector] = sc.parallelize(BArray, numPartitions).map(Vectors.dense)
    val b = new DenseVector(bArray)

    val (objective, allocation) = LP.solve(c, rows, b, solver = NewtonSolver.ConjugateGradient)
    val expectedSol = Vectors.dense(
      Array(1.66666667, 5.83333333, 40.0, 0.0, 0.0, 13.33333333, 9.16666667))
    val actualAllocation = Vectors.dense(allocation.flatMap(_.toArray).collect())

    assert(actualAllocation ~== expectedSol absTol 1e-6, "matrix-free LP.solve x should return the correct answer.")
    assert(objective ~== 12.083333333 absTol 1e-6, "matrix-free LP.solve should reach the same optimum.")
  }

  test("automatic CG preconditioner rank obeys its storage budget") {
    val millionRows = 1000000
    val tenMillionRows = 10000000

    val rankWithStorage = newton.CgFactory.autoMaxRank(millionRows)
    val rankWithoutStorage = newton.CgFactory.autoMaxRank(tenMillionRows)

    assert(rankWithStorage == 8)
    assert(rankWithoutStorage == 0)
  }

  test("capped normal-equations weights keep D squared equal to D2") {
    val x = sc.parallelize(Array(0.5, 4.0), numPartitions).glom.map(new DenseVector(_))
    val s = sc.parallelize(Array(0.25, 0.5), numPartitions).glom.map(new DenseVector(_))

    val weights = LP.normalEquationWeights(x, s, eps = 1.0, valueCap = 100.0)
    val d = weights.sqrt.flatMap(_.toArray).collect()
    val d2 = weights.squared.flatMap(_.toArray).collect()

    assert(Vectors.dense(d2) ~== Vectors.dense(50.0, 400.0) absTol 1e-12)
    assert(Vectors.dense(d.map(value => value * value)) ~== Vectors.dense(d2) absTol 1e-12)
  }

  test("full Cholesky factor serves multiple right-hand sides") {
    val packed = Array(4.0, 1.0, 3.0)
    val factor = CholeskyDecomposition.factor(packed, 2)

    val first = CholeskyDecomposition.solveFactored(factor, 2, Array(1.0, 2.0))
    val second = CholeskyDecomposition.solveFactored(factor, 2, Array(4.0, 3.0))

    assert(factor.length == 4)
    assert(packed.sameElements(Array(4.0, 1.0, 3.0)))
    assert(Vectors.dense(first) ~== Vectors.dense(1.0 / 11.0, 7.0 / 11.0) absTol 1e-12)
    assert(Vectors.dense(second) ~== Vectors.dense(9.0 / 11.0, 8.0 / 11.0) absTol 1e-12)
  }

  test("full Cholesky identifies a non-positive-definite leading minor") {
    val error = intercept[IllegalArgumentException](
      CholeskyDecomposition.factor(Array(1.0, 0.0, -1.0), 2))

    assert(error.getMessage.contains("not positive definite"))
    assert(error.getMessage.contains("leading minor 2"))
  }

  test("core Auto resolves from the known equality-form row count") {
    assert(LP.resolveNewtonSolver(NewtonSolver.Auto, 10000) == NewtonSolver.Cholesky)
    assert(LP.resolveNewtonSolver(NewtonSolver.Auto, 10001) == NewtonSolver.ConjugateGradient)
    assert(LP.resolveNewtonSolver(NewtonSolver.Auto, NewtonSolver.AutoCholeskyLimit) == NewtonSolver.Cholesky)
    assert(LP.resolveNewtonSolver(NewtonSolver.Auto, NewtonSolver.AutoCholeskyLimit + 1) ==
      NewtonSolver.ConjugateGradient)
  }

  test("primal infeasible: conflicting constraints terminate with a Farkas certificate") {
    implicit val s: SparkSession = spark
    // x <= 1 and x >= 2 in equality form: x + s1 = 1, x - s2 = 2 (distinct rows, full row rank)
    val cInf = sc.parallelize(Array(1.0, 0.0, 0.0), numPartitions).glom.map(new DenseVector(_))
    val atInf = sc.parallelize(Seq(
      Vectors.dense(1.0, 1.0), Vectors.dense(1.0, 0.0), Vectors.dense(0.0, -1.0)), numPartitions)
    val bInf = new DenseVector(Array(1.0, 2.0))

    val summary = LP.solveSummary(cInf, atInf, bInf)

    assert(summary.termination == LP.Termination.PrimalInfeasible)
    assert(summary.certificateResidual <= 1e-8)

    // the same certificate must be found by the matrix-free solver

    val cgSummary = LP.solveSummary(cInf, atInf, bInf, solver = NewtonSolver.ConjugateGradient)

    assert(cgSummary.termination == LP.Termination.PrimalInfeasible)
    assert(cgSummary.certificateResidual <= 1e-8)

    // the normalized ray y satisfies its Farkas inequalities: b^T y = 1, A^T y <= eps componentwise
    val y = summary.primalCertificate.get.toArray

    assert(math.abs(1.0 * y(0) + 2.0 * y(1) - 1.0) < 1e-9, "certificate is normalized to b^T y = 1")
    val aTy = Seq(y(0) + y(1), y(0), -y(1)) // columns x, s1, s2
    aTy.foreach(v => assert(v <= 1e-8, s"A^T y must be <= tolerance componentwise, got $aTy"))
  }

  test("dual infeasible (unbounded): diverging primal iterate terminates with a Farkas certificate") {
    implicit val s: SparkSession = spark
    // minimize -x1 subject to x1 - x2 = 1: feasible at (1, 0), unbounded along the ray (1, 1)
    val cUnb = sc.parallelize(Array(-1.0, 0.0), numPartitions).glom.map(new DenseVector(_))
    val atUnb = sc.parallelize(Seq(Vectors.dense(1.0), Vectors.dense(-1.0)), numPartitions)
    val bUnb = new DenseVector(Array(1.0))

    val summary = LP.solveSummary(cUnb, atUnb, bUnb)

    assert(summary.termination == LP.Termination.DualInfeasible)
    assert(summary.certificateResidual <= 1e-8)
    assert(summary.primalResidual < 1e-4, "the iterate stays near-feasible while diverging along the ray")

    // the normalized ray z satisfies its Farkas conditions: z >= 0, c^T z = -1, ||A z||_inf <= eps
    val z = summary.dualCertificate.get.flatMap(_.toArray).collect()

    assert(z.forall(_ >= 0.0), "certificate ray must be nonnegative")
    assert(math.abs(-z(0) + 1.0) < 1e-9, "certificate is normalized to c^T z = -1")
    assert(math.abs(z(0) - z(1)) <= 1e-8, "||A z||_inf must be <= tolerance")
  }
}
