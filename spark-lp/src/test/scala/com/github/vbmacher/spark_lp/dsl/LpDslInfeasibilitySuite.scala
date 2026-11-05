package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpDslInfeasibilitySuite extends AnyFunSuite with DataFrameSuiteBase {

  test("Infeasible: conflicting inequalities yield a certificate-backed status with NaN objective") {
    implicit val ss: SparkSession = spark
    // x <= 1 and x >= 2: distinct rows after slack introduction, so presolve does not catch it
    val model = LpProblem("infeasible", Minimize)
    val x = model.variable("x")
    model += 1.0 * x
    model += (x <= 1.0).named("cap")
    model += (x >= 2.0).named("floor")

    val solution = model.solve()

    assert(solution.status == LpStatus.Infeasible)
    val proof = solution.evidence.get.asInstanceOf[InfeasibilityCertificate]

    assert(proof.verify(1e-6).valid)
    assert(!proof.copy(rows = proof.rows.map(_ * -1.0)).verify(1e-6).valid)
    assert(solution.objectiveValue.isNaN)
    assert(solution.iterations > 0)

    // slack diagnostics on the last iterate locate the conflict: the two rows cannot both hold,
    // so their slacks sum to -1 and at least one is materially negative
    val slacks = solution.constraints.select("name", "slack").collect()
      .map(r => r.getString(0) -> r.getDouble(1)).toMap

    assert(slacks.keySet == Set("cap", "floor"))
    assert(math.abs(slacks("cap") + slacks("floor") + 1.0) < 1e-6)
    assert(slacks.values.min < -0.4)
  }

  test("Unbounded: dual certificate plus a primal-feasible iterate maps to +Infinity for Maximize") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("unbounded", Maximize)
    val x = model.variable("x")
    model += 1.0 * x
    model += (x >= 1.0).named("floor")

    val solution = model.solve(SolveConfig(tolerance = 1e-4))

    assert(solution.status == LpStatus.Unbounded)
    val proof = solution.evidence.get.asInstanceOf[UnboundedDirection]

    assert(proof.verify(1e-4).valid)
    assert(!proof.copy(direction = proof.direction.mapValues(-_)).verify(1e-4).valid)
    assert(solution.objectiveValue == Double.PositiveInfinity)
    assert(!solution.residuals.primal.isNaN)
  }

  test("Unbounded: signed infinity respects the objective sense (-Infinity for Minimize)") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("unbounded_min", Minimize)
    val x = model.variable("x")
    model += -1.0 * x
    model += (x >= 1.0).named("floor")

    val solution = model.solve(SolveConfig(tolerance = 1e-4))

    assert(solution.status == LpStatus.Unbounded)
    val proof = solution.evidence.get.asInstanceOf[UnboundedDirection]

    assert(proof.verify(1e-4).valid)
    assert(!proof.copy(direction = proof.direction.mapValues(-_)).verify(1e-4).valid)
    assert(solution.objectiveValue == Double.NegativeInfinity)
  }

  test("InfeasibleOrUnbounded: dual certificate without a primal-feasible point within tolerance") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("unbounded_strict", Maximize)
    val x = model.variable("x")
    model += 1.0 * x
    model += (x >= 1.0).named("floor")

    // at the default tolerance the last iterate does not qualify as primal-feasible, so the
    // dual-infeasibility certificate alone cannot distinguish infeasible from unbounded

    val solution = model.solve()

    assert(solution.status == LpStatus.InfeasibleOrUnbounded)
    val proof = solution.evidence.get.asInstanceOf[UnboundedDirection]

    assert(proof.point.isEmpty)
    assert(!proof.verify().valid)
    assert(solution.objectiveValue.isNaN)
  }

  test("an infeasible instance that previously threw LpNumericalException is reported as Infeasible") {
    implicit val ss: SparkSession = spark
    // without certificate detection this instance degenerates the Cholesky step and dies with
    // LpNumericalException; the certificate tests reclassify it truthfully
    val model = LpProblem("reclassified", Maximize)
    val w = model.variable("w")
    val z = model.variable("z")
    model += w + z
    model += (w + z === -1.0).named("impossible")
    model += (w === 5.0).named("pin")

    val solution = model.solve()

    assert(solution.status == LpStatus.Infeasible)
    val proof = solution.evidence.get.asInstanceOf[InfeasibilityCertificate]

    assert(proof.verify(1e-6).valid)
    assert(!proof.copy(rows = proof.rows.map(_ * -1.0)).verify(1e-6).valid)
    assert(solution.objectiveValue.isNaN)

    // the violated row is visible in the diagnostics
    val slacks = solution.constraints.select("name", "slack").collect()
      .map(r => r.getString(0) -> r.getDouble(1)).toMap

    assert(slacks("impossible") < -0.4)
  }
  test("certificates retain scaled rows, shifts, fixed values and explicit upper bounds") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("bounds")
    val x = model.variable("x", lowerBound = -3.0, upperBound = Some(2.0))
    val fixed = model.variable("fixed", lowerBound = 5.0, upperBound = Some(5.0))
    model += x + fixed
    model += (7.0 * x + fixed >= 26.0).named("scaled floor")

    val solution = model.solve()
    try {
      assert(solution.status == LpStatus.Infeasible)
      val proof = solution.evidence.get.asInstanceOf[InfeasibilityCertificate]

      assert(proof.model.rows.head.name == "scaled floor")
      assert(proof.verify(1e-6).valid)
      assert(!proof.copy(bounds = proof.bounds.mapValues { case (l, _) => (l, 0.0) }).verify(1e-6).valid)
    } finally solution.close()
  }

  test("shifted and free-variable rays omit shifts and give fixed variables zero direction") {
    implicit val ss: SparkSession = spark
    for (sense <- Seq(Minimize, Maximize); lower <- Seq(-4.0, Double.NegativeInfinity)) {
      val model = LpProblem("transformed ray", sense)
      val x = model.variable("x", lowerBound = lower)
      val fixed = model.variable("fixed", lowerBound = 3.0, upperBound = Some(3.0))
      model += (if (sense == Minimize) -1.0 * x + fixed else x + fixed)
      model += (x >= 1.0).named("floor")

      val solution = model.solve(SolveConfig(tolerance = 1e-4))
      try {
        val proof = solution.evidence.get.asInstanceOf[UnboundedDirection]
        if (solution.status == LpStatus.Unbounded) assert(proof.verify(1e-4).valid)
        else {
          assert(solution.status == LpStatus.InfeasibleOrUnbounded)
          assert(proof.point.isEmpty)
          val witness = proof.model.variables.mapValues(v => if (v.name == "fixed") 3.0 else 1.0)
          assert(proof.copy(point = Some(witness)).verify(1e-4).valid)
        }
        val fixedKeys = proof.model.variables.filter(_._2.name == "fixed").keys.collect().toSet

        assert(proof.direction.filter(x => fixedKeys(x._1)).values.collect().forall(_ == 0.0))
      } finally solution.close()
    }
  }

  test("independent verifier reports missing, nonfinite and tolerance-sensitive evidence") {
    val key = (0, "x")
    val original = EvidenceModel(Vector(EvidenceRow("impossible", None, "==", -1.0)),
      sc.parallelize(Seq(key -> EvidenceVariable("x", 0.0, None, 1.0, Map(0 -> 1.0)))), Minimize)
    val proof = InfeasibilityCertificate(original, Vector(-1.0), sc.parallelize(Seq(key -> (1.0, 0.0))))

    assert(proof.verify().valid)
    assert(!proof.copy(bounds = sc.emptyRDD[((Int, String), (Double, Double))]).verify().valid)
    assert(!proof.copy(bounds = sc.parallelize(Seq(key -> (Double.NaN, 0.0)))).verify().valid)
    assert(!proof.copy(rows = Vector(-1.0 + 1e-7)).verify(1e-9).valid)
    assert(proof.copy(rows = Vector(-1.0 + 1e-7)).verify(1e-6).valid)
  }

  test("independent QP ray verification requires zero curvature along the direction") {
    val key = (0, "x")
    val model = EvidenceModel(Vector.empty,
      sc.parallelize(Seq(key -> EvidenceVariable("x", 0.0, None, -1.0, Map.empty, curvature = 2.0))), Minimize)
    val direction = sc.parallelize(Seq(key -> 1.0))
    val point = sc.parallelize(Seq(key -> 0.0))

    assert(!UnboundedDirection(model, direction, Some(point)).verify().valid)
    val linear = model.copy(variables = model.variables.mapValues(_.copy(curvature = 0.0)))

    assert(UnboundedDirection(linear, direction, Some(point)).verify().valid)
  }

}
