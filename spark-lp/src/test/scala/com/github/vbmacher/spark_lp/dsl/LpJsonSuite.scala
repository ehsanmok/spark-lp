package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import java.nio.file.Files
import scala.collection.JavaConverters._
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.util.Try

class LpJsonSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("JSON preserves model algebra and separately marked solution metadata") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("json \"model\"", Maximize)
    val xs = model.variablesOf("items", Seq(("a", 1), ("b", 2)).toDS(), (x: (String, Int)) => x, category = Binary)
    val free = model.variable("free", Double.NegativeInfinity)
    val fixed = model.variable("fixed", 2.0, Some(2.0))
    model += xs.sum + free + fixed + 0.1234567890123456
    model += (xs.sum <= 1.5)
    model += (free === -1.0)

    val original = model.solve()
    val directory = Files.createTempDirectory("spark-lp-json-").resolve("model").toString
    try {
      LpJson.write(model, directory, Some(LpSolutionData.fromSolution(original)))

      intercept[LpModelException](LpJson.write(model, directory))
      val document = LpJson.read(directory)

      assert(!document.solutionMetadataVerified)
      assert(document.model.name == model.name)
      assert(document.model.objective.constant == 0.1234567890123456)
      assert(document.solution.get.residuals.nonEmpty)
      val imported = document.model.toProblem()

      val result = imported.model.solve()
      try {
        assert(math.abs(result.objectiveValue - 2.1234567890123456) < 1e-6)
        assert(result.status == LpStatus.Optimal)

        val report = imported.model.validateCandidate(document.solution.get.values.get)
        try {
          assert(report.feasible)
        } finally report.close()
      } finally result.close()
      LpJson.write(model, directory, overwrite = true)

      assert(LpJson.read(directory).solution.isEmpty)
    } finally {
      original.close()
      val path = new org.apache.hadoop.fs.Path(directory)
      path.getFileSystem(spark.sparkContext.hadoopConfiguration).delete(path.getParent, true)
    }
  }

  test("invalid schema and malformed categories are rejected by the JSON reader") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("schema")
    model.variable("x")
    val directory = Files.createTempDirectory("spark-lp-json-invalid-").resolve("model").toString
    try {
      LpJson.write(model, directory)
      val header = Files.list(java.nio.file.Paths.get(directory, "header"))
      val file = try header.iterator().asScala.find(_.getFileName.toString.startsWith("part-")).get finally header.close()
      val original = new String(Files.readAllBytes(file), java.nio.charset.StandardCharsets.UTF_8)
      // Replace through Hadoop so its checksum sidecar stays consistent.
      def replace(text: String): Unit = {
        val path = new org.apache.hadoop.fs.Path(file.toString)
        val out = path.getFileSystem(spark.sparkContext.hadoopConfiguration).create(path, true)
        try out.write(text.getBytes(java.nio.charset.StandardCharsets.UTF_8)) finally out.close()
      }
      replace(original.replace("\"schemaVersion\":1", "\"schemaVersion\":99"))

      intercept[LpModelException](LpJson.read(directory))
      replace(original.replace("\"Continuous\"", "\"InvalidCategory\""))

      intercept[LpModelException](LpJson.read(directory))
    } finally {
      val path = new org.apache.hadoop.fs.Path(directory)
      path.getFileSystem(spark.sparkContext.hadoopConfiguration).delete(path.getParent, true)
    }
  }

  test("no-overwrite publication preserves a destination created during the write") {
    implicit val ss: SparkSession = spark
    implicit val executionContext: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
    val model = LpProblem("concurrent publication")
    model.variable("x")
    val root = Files.createTempDirectory("spark-lp-json-race-")
    val directory = root.resolve("model")
    val writing = Future(Try(LpJson.write(model, directory.toString)))
    try {
      val deadline = 30.seconds.fromNow
      var temporarySeen = false
      while (!temporarySeen && !writing.isCompleted && deadline.hasTimeLeft()) {
        val children = Files.list(root)
        try temporarySeen = children.iterator().asScala.exists(_.getFileName.toString.startsWith("model.tmp-"))
        finally children.close()
        if (!temporarySeen) Thread.`yield`()
      }

      assert(temporarySeen, "writer did not expose its temporary publication directory")
      Files.createDirectories(directory)
      val marker = Files.write(directory.resolve("concurrent-marker"), Array[Byte](1))

      assert(Await.result(writing, 30.seconds).isFailure)
      assert(Files.exists(marker))
    } finally {
      Await.ready(writing, 30.seconds)
      val path = new org.apache.hadoop.fs.Path(root.toString)
      path.getFileSystem(spark.sparkContext.hadoopConfiguration).delete(path, true)
    }
  }
}
