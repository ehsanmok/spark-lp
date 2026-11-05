package com.github.vbmacher.spark_lp

import com.github.vbmacher.spark_lp.newton.NewtonSolver
import com.github.vbmacher.spark_lp.vectors.DMatrix
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpScalingSuite extends AnyFunSuite with DataFrameSuiteBase {
  for (solver <- Seq(NewtonSolver.Cholesky, NewtonSolver.ConjugateGradient)) {
    test(s"$solver retains infeasibility with an unrelated large RHS") {
      implicit val session: SparkSession = spark
      val c = sc.parallelize(Seq(0.0, 0.0), 1).glom().map(new DenseVector(_))
      val at: DMatrix = sc.parallelize(Seq(Vectors.dense(1.0, 0.0), Vectors.dense(0.0, 1.0)), 1)

      val result = LP.solveSummary(c, at, Vectors.dense(-1.0, 1e12).toDense, solver = solver)

      try {
        assert(result.termination == LP.Termination.PrimalInfeasible)
        assert(result.primalCertificate.nonEmpty)
      } finally result.x.unpersist(false)
    }

    test(s"$solver retains unbounded directions with independently pinned variables") {
      implicit val session: SparkSession = spark
      val c = sc.parallelize(Seq(-1.0, 0.0), 1).glom().map(new DenseVector(_))
      val at: DMatrix = sc.parallelize(Seq(Vectors.dense(0.0), Vectors.dense(1.0)), 1)

      val result = LP.solveSummary(c, at, Vectors.dense(1.0).toDense, solver = solver)

      try {
        val ray = result.dualCertificate.get.collect().flatMap(_.values)
        assert(result.termination == LP.Termination.DualInfeasible)
        assert(math.abs(ray(0) - 1.0) <= 1e-8)
        assert(math.abs(ray(1)) <= 1e-8)
      } finally {
        result.x.unpersist(false)
        result.dualCertificate.foreach(_.unpersist(false))
      }
    }

    test(s"$solver solves bounded LPs with large objective coefficients") {
      implicit val session: SparkSession = spark
      val c = sc.parallelize(Seq(-1e10, -2e10), 1).glom().map(new DenseVector(_))
      val at: DMatrix = sc.parallelize(Seq(Vectors.dense(1.0), Vectors.dense(1.0)), 1)

      val result = LP.solveSummary(c, at, Vectors.dense(1.0).toDense, solver = solver)

      try {
        val values = result.x.collect().flatMap(_.values)
        assert(result.termination == LP.Termination.Converged)
        assert(result.candidate.feasible)
        assert(math.abs(values.sum - 1.0) <= 1e-6)
        assert(values.forall(_ >= 0.0))
        assert(math.abs(values(0)) <= 1e-6)
        assert(math.abs(values(1) - 1.0) <= 1e-6)
        assert(math.abs(result.objectiveValue / -2e10 - 1.0) <= 1e-8)
        assert(result.primalCertificate.isEmpty && result.dualCertificate.isEmpty)
      } finally result.x.unpersist(false)
    }

    test(s"$solver solves feasible LPs with disparate right-hand sides") {
      implicit val session: SparkSession = spark
      val c = sc.parallelize(Seq(0.0, 1.0), 1).glom().map(new DenseVector(_))
      val at: DMatrix = sc.parallelize(Seq(Vectors.dense(1.0, 0.0), Vectors.dense(0.0, 1.0)), 1)

      val result = LP.solveSummary(c, at, Vectors.dense(100.0, 1e12).toDense, solver = solver)

      try {
        val values = result.x.collect().flatMap(_.values)
        assert(result.termination == LP.Termination.Converged)
        assert(result.candidate.feasible)
        assert(math.abs(values(0) - 100.0) / 101.0 <= 1e-8)
        assert(math.abs(values(1) / 1e12 - 1.0) <= 1e-8)
        assert(math.abs(result.objectiveValue / 1e12 - 1.0) <= 1e-8)
        assert(result.primalCertificate.isEmpty && result.dualCertificate.isEmpty)
      } finally result.x.unpersist(false)
    }
  }
}
