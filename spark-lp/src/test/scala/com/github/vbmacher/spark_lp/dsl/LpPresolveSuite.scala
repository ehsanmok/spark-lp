package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpPresolveSuite extends AnyFunSuite with DataFrameSuiteBase {
  private val full = SolveConfig(presolve = PresolveConfig(effort = PresolveEffort.Full))
  private val off = SolveConfig(presolve = PresolveConfig(enabled = false))

  test("singleton equality fixing and coupled bound propagation retain LP/MIP objectives and offsets") {
    implicit val ss: SparkSession = spark
    for (category <- Seq(Continuous, Integer); sense <- Seq(Minimize, Maximize)) {
      val model = LpProblem("presolve bounds", sense)
      val fixed = model.variable("fixed_by_row", upperBound = Some(10.0), category = category)
      val x = model.variable("x", upperBound = Some(10.0), category = category)
      model += fixed + x + 5.0
      model += (fixed === 2.0).named("fixed_row")
      model += (x + fixed <= 5.0).named("capacity")

      val reduced = model.solve(full)
      val baseline = model.solve(off)
      try {
        assert(reduced.status == LpStatus.Optimal)
        assert(baseline.status == LpStatus.Optimal)
        assert(math.abs(reduced.objectiveValue - baseline.objectiveValue) < 1e-6)
        assert(math.abs(reduced.value(fixed) - 2.0) < 1e-6)
        assert(reduced.presolve.exists(s => s.fixedVariables == 1 && s.bounds.size == 2 && s.passes > 0))
        assert(baseline.presolve.exists(s => !s.enabled && s.bounds.isEmpty))
        assert(reduced.reducedCost(fixed).isEmpty)

        val validation = model.validateCandidate(model.candidateValues(Seq(fixed -> reduced.value(fixed), x -> reduced.value(x))))
        try {
          assert(validation.feasible)
        } finally validation.close()
      } finally { reduced.close(); baseline.close() }

      assert(fixed.upperBound.contains(10.0))
      assert(model.inspect.constraintDeclarations.size == 2)
    }
  }

  test("free zero-cost singleton columns reconstruct original values and original row activities") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("substitute", Maximize)
    val x = model.variable("x", upperBound = Some(2.0))
    val free = model.variable("free", Double.NegativeInfinity)
    model += lpSum(x) + 7.0
    model += (2.0 * free + 3.0 * x === 10.0).named("definition")

    val reduced = model.solve(full)
    val baseline = model.solve(off)
    try {
      assert(reduced.status == LpStatus.Optimal)
      assert(baseline.status == LpStatus.Optimal)
      assert(math.abs(reduced.objectiveValue - 9.0) < 1e-6)
      assert(math.abs(reduced.objectiveValue - baseline.objectiveValue) < 1e-6)
      assert(math.abs(reduced.value(free) - 2.0) < 1e-6)
      assert(reduced.presolve.get.substitutions == Vector(LpSubstitution(LpPortableModel.variable(free).id, "definition", 2.0, 10.0)))
      assert(reduced.constraints.first().getAs[Double]("activity") == 10.0)
      assert(reduced.constraints.first().isNullAt(6))
      assert(reduced.presolve.get.solverColumns < baseline.presolve.get.solverColumns)
    } finally { reduced.close(); baseline.close() }
  }

  test("full presolve detects contradictions and preserves near-dependent distinct rows") {
    implicit val ss: SparkSession = spark
    val infeasible = LpProblem("contradiction")
    val x = infeasible.variable("x", upperBound = Some(1.0))
    infeasible += lpSum(x)
    infeasible += (x >= 2.0).named("contradiction")

    val impossible = infeasible.solve(full)
    try {
      assert(impossible.status == LpStatus.Infeasible)
      assert(!impossible.candidate.available)
    } finally impossible.close()
    val model = LpProblem("distinct")
    val a = model.variable("a", upperBound = Some(2.0))
    val b = model.variable("b", upperBound = Some(2.0))
    model += a + b
    model += (a + b === 2.0).named("first")
    model += (a + 1.0001 * b === 2.0001).named("second")

    val reduced = model.solve(full)
    try {
      assert(reduced.status == LpStatus.Optimal)
      assert(reduced.constraints.count() == 2)
      assert(reduced.presolve.get.substitutions.isEmpty)
      assert(reduced.presolve.get.reducedRows == 2)
      assert(math.abs(reduced.value(a) - 1.0) < 1e-4)
      assert(math.abs(reduced.value(b) - 1.0) < 1e-4)
    } finally reduced.close()
  }

  test("substitution reconstructs values while the remaining coupled model is optimized") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("partial substitution")
    val x = model.variable("x", upperBound = Some(0.8))
    val y = model.variable("y", upperBound = Some(1.0))
    val free = model.variable("free", Double.NegativeInfinity)
    model += 2.0 * x + y
    model += (x + y >= 1.0).named("minimum")
    model += (x + y + free === 3.0).named("definition")

    val result = model.solve(full)
    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.objectiveValue - 1.0) < 1e-6)
      assert(result.presolve.exists(s => s.substitutions.size == 1 && s.solverRows > 0))
      assert(math.abs(result.value(free) - 2.0) < 1e-6)
      assert(result.candidate.feasible)
    } finally result.close()
  }
}
