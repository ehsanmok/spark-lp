package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpInferredBoundsSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("free and one-sided integer variables acquire finite intervals from coupled constraints") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("inferred", Maximize)
    val x = model.variable("free_integer", Double.NegativeInfinity, category = Integer)
    val y = model.variable("one_sided", category = Integer)
    model += x + y + 5.0
    model += (x >= -2.2)
    model += (x <= 3.8)
    model += (y <= x + 2.0)

    val result = model.solve()
    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.objectiveValue - 13.0) < 1e-6)
      assert(result.value(x) == 3.0)
      assert(result.value(y) == 5.0)
      assert(x.lowerBound.isNegInfinity)
      assert(x.upperBound.isEmpty)
      assert(y.upperBound.isEmpty)

      val report = model.validateCandidate(model.candidateValues(Seq(x -> result.value(x), y -> result.value(y))))
      try {
        assert(report.feasible)
      } finally report.close()
    } finally result.close()

    val relaxation = model.solve(SolveConfig(relaxIntegrality = true))
    try {
      assert(math.abs(relaxation.objectiveValue - 14.6) < 1e-6)
    } finally relaxation.close()
  }

  test("keyed constraints infer distinct bounds without changing declared family metadata") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("keyed", Maximize)
    val xs = model.variables("x", Seq("a", "b").toDF("key"), $"key", category = Integer)
    model += xs.sum
    model += (xs.sumBy("key")() <= Seq(("a", 2.8), ("b", 4.2)).toDF("key", "rhs"))

    val result = model.solve()
    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.objectiveValue - 6.0) < 1e-6)
      assert(result.value(xs("a")) == 2.0)
      assert(result.value(xs("b")) == 4.0)
      assert(xs.upperBound.isEmpty)
    } finally result.close()
  }

  test("integer infeasibility, analytical unboundedness and unsupported search remain distinct") {
    implicit val ss: SparkSession = spark
    val impossible = LpProblem("integer contradiction")
    val x = impossible.variable("x", Double.NegativeInfinity, category = Integer)
    val free = impossible.variable("free", Double.NegativeInfinity)
    impossible += -1.0 * free
    impossible += (x === 0.5)

    val infeasible = impossible.solve()
    try {
      assert(infeasible.status == LpStatus.Infeasible)
      assert(!infeasible.candidate.available)
    } finally infeasible.close()

    val unbounded = LpProblem("rowless integer", Maximize)
    val z = unbounded.variable("z", category = Integer)
    unbounded += lpSum(z)

    val ray = unbounded.solve()
    try {
      assert(ray.status == LpStatus.Unbounded)
      assert(ray.objectiveValue.isPosInfinity)
      assert(ray.value(z) == 0.0)
      assert(ray.candidate.feasible)
      assert(ray.mip.exists(_.termination == "AnalyticalUnbounded"))
    } finally ray.close()

    val unresolved = LpProblem("unbounded LP does not prove MIP unbounded", Maximize)
    val a = unresolved.variable("a", Double.NegativeInfinity, category = Integer)
    val b = unresolved.variable("b", Double.NegativeInfinity, category = Integer)
    unresolved += lpSum(a)
    unresolved += (a - b === 0.5)

    val error = intercept[LpModelException](unresolved.solve())

    assert(error.getMessage.contains("could not infer a finite integer interval"))

    val lp = unresolved.solve(SolveConfig(relaxIntegrality = true))
    try {
      assert(lp.status == LpStatus.Unbounded || lp.status == LpStatus.InfeasibleOrUnbounded)
    } finally lp.close()
  }
}
