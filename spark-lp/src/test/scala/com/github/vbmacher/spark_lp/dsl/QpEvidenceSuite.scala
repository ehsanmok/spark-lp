package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class QpEvidenceSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("QP certificates retain original costs and curvature through bound shifts") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("qp infeasible")
    val x = model.variable("x", lowerBound = -3, upperBound = Some(2.0))
    model += QpObjective.squaredDeviation(x, 1.0)
    model += (x >= 3.0)

    val result = model.solve(SolveConfig(newtonSolver = NewtonSolver.ConjugateGradient))
    try {
      assert(result.status == LpStatus.Infeasible)
      val proof = result.evidence.get.asInstanceOf[InfeasibilityCertificate]

      assert(proof.verify(1e-6).valid)
      val v = proof.model.variables.values.collect().head

      assert(v.curvature == 2.0)
      assert(v.cost == -2.0)
      assert(v.lower == -3.0)
    } finally result.close()
  }

  test("coupled QP infeasibility evidence verifies the factor-expanded model") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("coupled infeasible")
    val x = model.variable("x")
    val y = model.variable("y")
    model += QpObjective.squared(x + y - 1.0)
    model += (x <= 1.0)
    model += (x >= 2.0)

    val result = model.solve(SolveConfig(newtonSolver = NewtonSolver.ConjugateGradient))
    try {
      assert(result.status == LpStatus.Infeasible)
      val proof = result.evidence.get.asInstanceOf[InfeasibilityCertificate]

      assert(proof.verify(1e-5).valid)
      assert(proof.model.variables.values.filter(_.curvature > 0.0).count() == 2)
    } finally result.close()
  }

  test("QP recession evidence can be checked against a separate original-model feasible point") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("flat ray")
    val x = model.variable("x")
    val y = model.variable("y")
    model += QpObjective.separable(2.0 * x, -1.0 * y)
    model += (x === 0.0)
    model += (y >= 1.0)

    val result = model.solve(SolveConfig(tolerance = 1e-4))
    try {
      val proof = result.evidence.get.asInstanceOf[UnboundedDirection]
      val point = proof.model.variables.mapValues(v => if (v.name == "x") 0.0 else 1.0)
      assert(proof.copy(point = Some(point)).verify(1e-4).valid)
      val altered = proof.direction.join(proof.model.variables).mapValues { case (d, v) =>
        if (v.name == "x") 1.0 else d
      }

      assert(!proof.copy(direction = altered, point = Some(point)).verify(1e-4).valid)
    } finally result.close()
  }
}
