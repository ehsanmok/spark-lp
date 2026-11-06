package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import java.nio.file.{Files, Path}
import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}

private[dsl] object HighsInterop {
  def available: Boolean = new ProcessBuilder("python3", "-c", "from scipy.optimize._highspy._core import _Highs")
    .redirectErrorStream(true).start().waitFor() == 0
  def inspect(path: Path): JsonNode = {
    val code = """import sys,json,math
from scipy.optimize._highspy._core import _Highs,HighsStatus
h=_Highs(); h.setOptionValue('output_flag',False)
status=h.readModel(sys.argv[1])
assert status==HighsStatus.kOk, str(status)
h.run(); p=h.getLp()
print(json.dumps(dict(columns=p.num_col_,rows=p.num_row_,offset=p.offset_,
 objective=h.getInfo().objective_function_value,status=str(h.getModelStatus()),
 lower=[v if math.isfinite(v) else str(v) for v in p.col_lower_],
 upper=[v if math.isfinite(v) else str(v) for v in p.col_upper_],cost=list(p.col_cost_),
 names=list(p.col_names_),integers=[int(v) for v in p.integrality_])))
"""
    val process = new ProcessBuilder("python3", "-c", code, path.toString).redirectErrorStream(true).start()
    val output = scala.io.Source.fromInputStream(process.getInputStream)
    val text = try output.mkString finally output.close()
    require(process.waitFor() == 0, text)
    new ObjectMapper().readTree(text.linesIterator.filter(_.startsWith("{")).toVector.last)
  }
}

class LpExportSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("independent HiGHS reader preserves LP bounds, integrality, sense and constants") {
    if (!HighsInterop.available) cancel("Optional installed SciPy/HiGHS reader unavailable")
    implicit val ss: SparkSession = spark
    val model = LpProblem("LP export", Maximize)
    val x = model.variable("unusual x[]", Double.NegativeInfinity, Some(2.0))
    val free = model.variable("free x", Double.NegativeInfinity)
    val g = model.variable("integer", upperBound = Some(3.0), category = Integer)
    val b = model.variable("binary", category = Binary)
    model.variable("fixed", 2.0, Some(2.0))
    model += 3.0 * x + free - g - b + 7.0
    model += (x >= -1.0).named("lower x")
    model += (free === -1.0).named("free equation")
    model += (g >= 0.2)
    model += (b >= 0.2)
    val directory = Files.createTempDirectory("spark-lp-export-")
    val first = directory.resolve("model.lp")
    val second = directory.resolve("second.lp")
    val mapping = LpExport.lp(model.inspect, first)
    val again = LpExport.lp(model.inspect, second)
    try {
      assert(java.util.Arrays.equals(Files.readAllBytes(first), Files.readAllBytes(second)))
      val parsed = HighsInterop.inspect(first)

      assert(parsed.get("columns").asInt() == 5)
      assert(parsed.get("rows").asInt() == 4)
      assert(parsed.get("offset").asDouble() == 7.0)
      assert(parsed.get("lower").get(0).asText() == "-inf")
      assert(parsed.get("upper").get(0).asDouble() == 2.0)
      assert(parsed.get("integers").get(2).asInt() == 1)
      assert(parsed.get("integers").get(3).asInt() == 1)
      assert(math.abs(parsed.get("objective").asDouble() - 10.0) < 1e-8)
      assert(parsed.get("status").asText().contains("Optimal"))
      assert(mapping.variables.count() == 5)
      assert(mapping.constraints.count() == 4)
      intercept[LpModelException](LpExport.lp(model.inspect, first))
      intercept[LpModelException](LpExport.lp(model.inspect, directory.resolve("bad.lp"), ExportNaming.Original))
      assert(!Files.exists(directory.resolve("bad.lp")))
      assert(model.handles.size == 5)
    } finally { mapping.close(); again.close(); Files.deleteIfExists(first); Files.deleteIfExists(second); Files.deleteIfExists(directory) }
  }
  test("continuous composite groups, empty rows and strict original names export explicitly") {
    if (!HighsInterop.available) cancel("Optional installed SciPy/HiGHS reader unavailable")
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("groups")
    val xs = model.variables("x", Seq(("a", 1), ("b", 2)).toDF("g", "i"),
      org.apache.spark.sql.functions.struct($"g", $"i"))
    model += xs.sum + xs.sum - xs.sum + 2.0
    model += (xs.sumBy("g")() >= Seq(("a", 1.0), ("b", 2.0), ("c", 0.0)).toDF("g", "rhs")).named("demand")
    val file = Files.createTempDirectory("spark-lp-groups-").resolve("model.lp")
    val mapping = LpExport.lp(model.inspect, file)
    try {
      val parsed = HighsInterop.inspect(file)

      assert(parsed.get("rows").asInt() == 3)
      assert(parsed.get("columns").asInt() == 2)
      assert(math.abs(parsed.get("objective").asDouble() - 5.0) < 1e-8)
      assert(mapping.variables.collect().map(_.originalName).toSet == Set("x[a,1]", "x[b,2]"))
    } finally { mapping.close(); Files.delete(file); Files.delete(file.getParent) }
    val safe = LpProblem("safe")
    val x = safe.variable("amount", upperBound = Some(2.0))
    safe += (x >= 1.0).named("minimum_amount")
    val original = Files.createTempDirectory("spark-lp-names-").resolve("model.lp")
    val names = LpExport.lp(safe.inspect, original, ExportNaming.Original)
    try {
      assert(HighsInterop.inspect(original).get("objective").asDouble() == 0.0)
      assert(names.variables.first().exportedName == "amount")
    } finally { names.close(); Files.delete(original); Files.delete(original.getParent) }
  }

}
