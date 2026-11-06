package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import java.nio.file.Files

class LpMpsSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("HiGHS reads free MPS objective sense, offsets, explicit bounds and integer markers") {
    if (!HighsInterop.available) cancel("Optional installed SciPy/HiGHS reader unavailable")
    implicit val ss: SparkSession = spark
    val model = LpProblem("mps", Maximize)
    val x = model.variable("amount", upperBound = Some(3.0), category = Integer)
    val b = model.variable("choice", category = Binary)
    val free = model.variable("free_amount", Double.NegativeInfinity)
    val upper = model.variable("upper_only", Double.NegativeInfinity, Some(2.0))
    model.variable("fixed", 2.0, Some(2.0))
    model += x + b + free + upper + 9.0
    model += (x <= 1.5).named("capacity")
    model += (b <= 0.5).named("choice_limit")
    model += (free === -1.0).named("free_value")
    val dir = Files.createTempDirectory("spark-lp-mps-")
    val mip = dir.resolve("integer.mps")
    val relaxation = dir.resolve("relaxation.mps")
    val normalized = dir.resolve("normalized.mps")
    val mappings = Vector(
      LpMps.write(model.inspect, mip, ExportNaming.Original),
      LpMps.write(model.inspect, relaxation, ExportNaming.Original, relaxIntegrality = true),
      LpMps.write(model.inspect, normalized))
    try {
      val integer = HighsInterop.inspect(mip)
      val relaxed = HighsInterop.inspect(relaxation)

      assert(integer.get("columns").asInt() == 5)
      assert(integer.get("rows").asInt() == 3)
      assert(integer.get("offset").asDouble() == 9.0)
      assert(math.abs(integer.get("objective").asDouble() - 11.0) < 1e-8)
      assert(math.abs(relaxed.get("objective").asDouble() - 12.0) < 1e-8)
      assert(integer.get("integers").toString.contains("1"))
      assert(!relaxed.get("integers").toString.contains("1"))
      assert(HighsInterop.inspect(normalized).get("objective").asDouble() == 11.0)
      assert(mappings.head.variables.collect().forall(v => v.originalName == v.exportedName))
      intercept[LpModelException](LpMps.write(model.inspect, mip))
    } finally {
      mappings.foreach(_.close())
      Vector(mip, relaxation, normalized).foreach(Files.delete)
      Files.delete(dir)
    }
  }
}
