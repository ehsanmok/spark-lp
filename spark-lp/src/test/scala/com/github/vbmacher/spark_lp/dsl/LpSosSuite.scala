package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import java.nio.file.Files

class LpSosSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("SOS1 excludes a fractional two-member assignment and preserves copied identities") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("selection", Maximize)
    val x = model.variable("x", upperBound = Some(1.0))
    val y = model.variable("y", upperBound = Some(1.0))
    model += x + y
    model += (x === y).named("balance")
    model.addSos1("one_nonzero", Seq(x -> 0.0, y -> 0.0))

    val integer = model.solve()
    val relaxation = model.solve(SolveConfig(relaxIntegrality = true))
    try {
      assert(integer.status == LpStatus.Optimal)
      assert(math.abs(integer.objectiveValue) < 1e-6)
      assert(integer.mip.nonEmpty)
      assert(integer.candidate.feasible)
      assert(math.abs(relaxation.objectiveValue - 1.0) < 1e-6)

      val invalid = model.validateCandidate(model.candidateValues(Seq(x -> relaxation.value(x), y -> relaxation.value(y))))
      try {
        assert(!invalid.feasible && invalid.violations.filter(_.kind == "sos:one_nonzero").first().magnitude == 1.0)
      } finally invalid.close()
      val copied = model.copy()

      assert(copied.model.inspect.sosGroups.size == 1)

      val result = copied.model.solve()
      try {
        assert(math.abs(result.value(copied.variable(x))) < 1e-6)
      } finally result.close()
    } finally { integer.close(); relaxation.close() }
  }

  test("keyed SOS2 enforces adjacent interpolation weights and survives JSON round trips") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("interpolation")
    val xs = model.variables("lambda", Seq(0, 1, 2).toDF("key"), $"key", upperBound = Some(1.0))
    model += lpSum(xs(1))
    model += (xs.sum === 1.0).named("convexity")
    model += (xs(1) + 2.0 * xs(2) === 1.0).named("position")
    model.addSos2("adjacent", Seq(xs(2) -> 2.0, xs(0) -> 0.0, xs(1) -> 1.0))

    val integer = model.solve()
    val relaxation = model.solve(SolveConfig(relaxIntegrality = true))
    val directory = Files.createTempDirectory("spark-lp-sos-")
    val json = directory.resolve("model")
    try {
      assert(integer.status == LpStatus.Optimal)
      assert(math.abs(integer.objectiveValue - 1.0) < 1e-6)
      assert(math.abs(integer.value(xs(1)) - 1.0) < 1e-6)
      assert(math.abs(relaxation.objectiveValue) < 1e-6)
      assert(LpSolutionData.fromSolution(integer).values.get.count() == 3)
      assert(LpAlgebra.preview(model.inspect).contains("Sos2"))
      LpJson.write(model, json.toString, Some(LpSolutionData.fromSolution(integer)))
      val document = LpJson.read(json.toString)
      val imported = document.model.toProblem()

      assert(imported.model.inspect.sosGroups.head.members.map(_.weight) == Vector(0.0, 1.0, 2.0))

      val result = imported.model.solve()
      try {
        assert(math.abs(result.objectiveValue - 1.0) < 1e-6)
      } finally result.close()

      intercept[LpModelException](LpExport.lp(model.inspect, directory.resolve("bad.lp")))
      intercept[LpModelException](LpMps.write(model.inspect, directory.resolve("bad.mps")))
      assert(!Files.exists(directory.resolve("bad.lp")))
    } finally {
      integer.close(); relaxation.close()
      val fs = new org.apache.hadoop.fs.Path(json.toString).getFileSystem(spark.sparkContext.hadoopConfiguration)
      fs.delete(new org.apache.hadoop.fs.Path(directory.toString), true)
    }
  }

  test("SOS validation handles tolerances, overlaps, integer/binary members and invalid declarations") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("mixed", Maximize)
    val i = model.variable("i", upperBound = Some(2.0), category = Integer)
    val b = model.variable("b", category = Binary)
    val x = model.variable("x", upperBound = Some(1.0))
    model += i + b + x
    model.addSos1("first", Seq(i -> 1.0, b -> 2.0))
    model.addSos1("overlap", Seq(b -> 1.0, x -> 2.0))

    val result = model.solve()
    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.objectiveValue - 3.0) < 1e-6)
    } finally result.close()
    val values = model.candidateValues(Seq(i -> 1.0, b -> 1e-8, x -> 0.0))

    val tolerant = model.validateCandidate(values)

    val strict = model.validateCandidate(values, CandidateValidationConfig(sosZeroTolerance = 0.0))
    try {
      assert(tolerant.feasible)
      assert(!strict.feasible) } finally { tolerant.close()
      strict.close()
    }

    intercept[LpModelException](model.addSos1("first", Seq(x -> 1.0)))
    intercept[LpModelException](model.addSos1("duplicate", Seq(x -> 1.0, x -> 2.0)))
    intercept[LpModelException](model.addSos2("ties", Seq(x -> 1.0, b -> 1.0)))
    intercept[LpModelException](model.addSos1("weight", Seq(x -> Double.NaN)))
    intercept[LpModelException](model.addSos1("foreign", Seq(LpProblem("other").variable("x") -> 1.0)))
    val unsupported = LpProblem("finite bounds")
    val free = unsupported.variable("free", Double.NegativeInfinity)
    unsupported.addSos1("group", Seq(free -> 1.0))

    assert(intercept[LpModelException](unsupported.solve()).getMessage.contains("finite declared bounds"))
  }
}
