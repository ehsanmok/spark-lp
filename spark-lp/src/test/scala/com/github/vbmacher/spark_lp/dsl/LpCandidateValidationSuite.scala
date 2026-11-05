package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpCandidateValidationSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("candidate validation reports objective values even for infeasible assignments") {
    implicit val ss: SparkSession = spark
    val cases = Seq(
      ("feasible", 2.0, 1.0, true, 10.0),
      ("fractional integer", 2.5, 0.0, false, 10.0),
      ("above upper bound", 5.0, -2.0, false, 13.0),
      ("violated constraint", 2.0, 2.0, false, 11.0))

    cases.foreach { case (scenario, xValue, yValue, expectedFeasible, expectedObjective) =>
      withClue(s"$scenario: ") {
        val model = LpProblem("candidate", Maximize)
        val x = model.variable("x", upperBound = Some(4.0), category = Integer)
        val y = model.variable("y", lowerBound = -2.0)
        model += 2.0 * x + y + 5.0
        model += (x + y <= 3.0).named("limit")
        val assignment = model.candidateValues(Seq(x -> xValue, y -> yValue))

        val report = model.validateCandidate(assignment)

        try {
          assert(report.feasible == expectedFeasible)
          assert(report.objectiveValue.contains(expectedObjective))
          assert(report.sense == Maximize)
        } finally report.close()
      }
    }
  }

  test("relaxed validation accepts fractional integers and marks the report as relaxation only") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("relaxed candidate", Maximize)
    val x = model.variable("x", upperBound = Some(4.0), category = Integer)
    val y = model.variable("y", lowerBound = -2.0)
    model += 2.0 * x + y + 5.0
    model += (x + y <= 3.0).named("limit")
    val assignment = model.candidateValues(Seq(x -> 2.5, y -> 0.0))

    val report = model.validateCandidate(assignment, CandidateValidationConfig(relaxIntegrality = true))

    try {
      assert(report.feasible)
      assert(report.relaxationOnly)
    } finally report.close()
  }

  test("default validation accepts an integer value within numerical tolerance") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("near integer", Maximize)
    val x = model.variable("x", upperBound = Some(4.0), category = Integer)
    val y = model.variable("y", lowerBound = -2.0)
    model += 2.0 * x + y + 5.0
    model += (x + y <= 3.0).named("limit")
    val assignment = model.candidateValues(Seq(x -> 2.0000001, y -> 0.0))

    val report = model.validateCandidate(assignment)

    try {
      assert(report.feasible)
    } finally report.close()
  }

  test("candidate assignments reject variables owned by another problem") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("candidate")
    val foreignVariable = LpProblem("foreign").variable("z")

    intercept[LpModelException](model.candidateValues(Seq(foreignVariable -> 0.0)))
  }

  test("malformed keyed assignments have no objective and report infinite violation") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("keys")
    val family = model.variables("x", Seq("a", "b").toDF("key"), $"key")
    model += family.sum
    val complete = model.candidateValues(Seq(family("a") -> 1.0, family("b") -> 2.0)).cache()
    try {
      val malformedAssignments = Seq(
        "missing member" -> complete.filter(_.value == 1.0),
        "duplicate members" -> complete.union(complete),
        "non-finite values" -> complete.map(value => value.copy(value = Double.NaN)),
        "foreign member" -> complete.union(spark.sparkContext.parallelize(Seq(
          LpCandidateValue(LpVariableId(99, "foreign"), 0.0)))))

      malformedAssignments.foreach { case (scenario, assignment) =>
        withClue(s"$scenario: ") {
          val report = model.validateCandidate(assignment)

          try {
            assert(!report.feasible)
            assert(report.objectiveValue.isEmpty)
            assert(report.maxViolation.isPosInfinity)
          } finally report.close()
        }
      }

      // A valid report shares caller input but must not release its cache.
      val report = model.validateCandidate(complete)

      try {
        assert(report.feasible)
        assert(report.objectiveValue.contains(3.0))
      } finally report.close()

      assert(complete.getStorageLevel.useMemory)
      assert(complete.count() == 2)
    } finally complete.unpersist()

    assert(spark.range(2).count() == 2)
  }

  test("quadratic candidate objectives retain curvature and sparse factor constants") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("qp")
    val x = model.variable("x")
    model += QpObjective.separable(2.0 * x, x + 3.0) + QpObjective.squared(x - 1.0, 2.0)
    val assignment = model.candidateValues(Seq(x -> 2.0))

    val report = model.validateCandidate(assignment)

    try {
      assert(report.feasible)
      assert(report.objectiveValue.contains(11.0))
    } finally report.close()
  }
}
