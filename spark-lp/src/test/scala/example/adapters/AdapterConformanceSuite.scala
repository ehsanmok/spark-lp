package example.adapters

import com.github.vbmacher.spark_lp.dsl._
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import java.nio.file.{Files, Path}
import scala.collection.JavaConverters._
import scala.concurrent.duration._

/** Application-owned adapter: exact separable bounded LP/MIP optimization, public API only. */
final class BoxAdapter extends LpSolverAdapter {
  override val name = "example-box"
  override val capabilities = LpSolverCapabilities(mip = true, infiniteBounds = false)
  var closed = false
  override def prepare(model: LpModelView, options: LpAdapterOptions): LpAdapterSession = {
    require(model.constraints.take(1).isEmpty, "BoxAdapter supports rowless models only")
    val variables = model.variables
    val costs = model.objectiveCoefficients.map(c => c.variable -> c.value)
    val sense = model.sense
    new LpAdapterSession {
      override def solve(): LpAdapterResult = {
        if (closed) throw new IllegalStateException("closed")
        if (options.shouldStop()) return LpAdapterResult(LpStatus.Stopped)
        val objectiveSign = if (sense == Minimize) 1.0 else -1.0
        val values = variables.map(v => v.id -> v).leftOuterJoin(costs).map { case (id, (v, cost)) =>
          val lower = if (v.category == Continuous) v.lower else math.ceil(math.max(v.lower, if (v.category == Binary) 0.0 else v.lower))
          val upper = if (v.category == Continuous) v.upper.get else math.floor(math.min(v.upper.get, if (v.category == Binary) 1.0 else v.upper.get))
          require(lower <= upper, "Empty box domain")
          val c = cost.getOrElse(0.0) * objectiveSign
          LpCandidateValue(id, if (c >= 0.0) lower else upper)
        }
        LpAdapterResult(LpStatus.Optimal, Some(values))
      }
      override def close(): Unit = closed = true
    }
  }
}

/** Optional independent CLI conformance fixture; SciPy is not a library dependency. */
final class PythonHighsAdapter(root: Path) extends LpCommandAdapter(LpCommandOptions(temporaryRoot = Some(root))) {
  override val name = "example-scipy-highs"
  override val capabilities = LpSolverCapabilities(mip = true)
  override protected def command(model: Path, output: Path, options: LpAdapterOptions): Seq[String] =
    Seq("python3", "-c", """import sys,json
from scipy.optimize._highspy._core import _Highs,HighsStatus
h=_Highs(); h.setOptionValue('output_flag',False)
assert h.readModel(sys.argv[1])==HighsStatus.kOk
assert h.run()==HighsStatus.kOk
assert 'Optimal' in str(h.getModelStatus())
p=h.getLp(); s=h.getSolution()
with open(sys.argv[2],'w') as f: json.dump({'objective':h.getInfo().objective_function_value,'values':dict(zip(p.col_names_,s.col_value))},f)
""", model.toString, output.toString)
  override protected def parse(path: Path, mapping: LpExportMapping, options: LpAdapterOptions): LpAdapterResult = {
    val json = new com.fasterxml.jackson.databind.ObjectMapper().readTree(path.toFile)
    val pairs = json.get("values").fields().asScala.map(e => e.getKey -> e.getValue.asDouble()).toMap
    val values = mapping.variables.map(v => LpCandidateValue(v.id, pairs(v.exportedName)))
    LpAdapterResult(LpStatus.Optimal, Some(values), Some(json.get("objective").asDouble()))
  }
}

class AdapterConformanceSuite extends AnyFunSuite with DataFrameSuiteBase {
  private def empty(path: Path): Boolean = { val s = Files.list(path); try !s.findAny().isPresent finally s.close() }

  test("external in-process adapter preserves keyed identities, sense, constants and caller caches") {
    implicit val ss: SparkSession = spark
    val local = spark
    import local.implicits._
    for (sense <- Seq(Minimize, Maximize); category <- Seq(Continuous, Integer)) {
      val input = Seq("a", "b").toDF("key").cache()
      input.count()
      val model = LpProblem("box", sense)
      val x = model.variables("x", input, $"key", lowerBound = -1.4, upperBound = Some(2.8), category = category)
      model += x("a") - 2.0 * x("b") + 7.0
      val adapter = new BoxAdapter

      val result = model.solve(adapter)
      val builtIn = model.solve()
      try {
        assert(result.status == LpStatus.Optimal)
        assert(result.candidate.feasible)
        assert(math.abs(result.objectiveValue - builtIn.objectiveValue) < 1e-8)
        assert(result.value(x("a")) == builtIn.value(x("a")))
        assert(adapter.closed)
        assert(result.backend.exists(_.independentlyValidated))
        assert(result.values(x).count() == 2)
        assert(input.storageLevel.useMemory)
      } finally { result.close(); builtIn.close(); input.unpersist() }
    }
  }

  test("real CLI solver maps continuous and mixed models and cleans its artifacts") {
    val probe = new ProcessBuilder("python3", "-c", "from scipy.optimize._highspy._core import _Highs").start()
    if (probe.waitFor() != 0) cancel("Optional SciPy HiGHS installation unavailable")
    implicit val ss: SparkSession = spark
    val root = Files.createTempDirectory("adapter path with spaces ")
    try for (category <- Seq(Continuous, Integer)) {
      val model = LpProblem("cli", Maximize)
      val x = model.variable("upper only", Double.NegativeInfinity, Some(3.0))
      val y = model.variable("integer", upperBound = Some(3.0), category = category)
      model += 2.0 * x + y + 7.0
      model += (x + y <= 3.5).named("capacity")

      val result = model.solve(new PythonHighsAdapter(root))
      try {
        assert(result.status == LpStatus.Optimal)
        assert(result.candidate.feasible)
        assert(math.abs(result.objectiveValue - (if (category == Integer) 13.0 else 13.5)) < 1e-8)
        assert(result.constraints.first().getAs[String]("name") == "capacity")
        assert(empty(root))
      } finally result.close()
    } finally Files.delete(root)
  }

  test("capability checks and malformed results fail explicitly with exception cleanup") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("validation")
    val x = model.variable("x", upperBound = Some(1.0))
    model += x + 3.0
    val id = model.inspect.variables.first().id
    var closedCount = 0
    def adapter(result: => LpAdapterResult, caps: LpSolverCapabilities = LpSolverCapabilities()): LpSolverAdapter =
      new LpSolverAdapter {
        override val name = "fixture"
        override val capabilities = caps
        override def prepare(view: LpModelView, options: LpAdapterOptions): LpAdapterSession = new LpAdapterSession {
          override def solve(): LpAdapterResult = result
          override def close(): Unit = closedCount += 1
        }
      }
    val badValues = Seq(Seq(LpCandidateValue(id, Double.NaN)), Seq.empty,
      Seq(LpCandidateValue(id, 0.0), LpCandidateValue(id, 0.0)), Seq(LpCandidateValue(LpVariableId(999, ""), 0.0)))
    badValues.foreach { records =>
      intercept[LpModelException](model.solve(adapter(LpAdapterResult(LpStatus.Optimal, Some(sc.parallelize(records))))))
    }
    val good = Some(model.candidateValues(Seq(x -> 0.0)))

    intercept[LpModelException](model.solve(adapter(LpAdapterResult(LpStatus.Optimal, good, Some(-3.0)))))
    intercept[LpModelException](model.solve(adapter(LpAdapterResult(LpStatus.Optimal, good, bestBound = Some(4.0)))))
    intercept[LpModelException](model.solve(adapter(LpAdapterResult(LpStatus.Optimal))))
    intercept[IllegalStateException](model.solve(adapter(throw new IllegalStateException("backend failure"))))
    assert(closedCount == 8)
    intercept[LpModelException](model.solve(adapter(LpAdapterResult(LpStatus.Stopped)), LpAdapterOptions(requireDuals = true)))
    intercept[LpModelException](model.solve(adapter(LpAdapterResult(LpStatus.Stopped)), LpAdapterOptions(onProgress = Some(_ => ()))))
    intercept[LpModelException](model.solve(adapter(LpAdapterResult(LpStatus.Stopped)), LpAdapterOptions(maxVariables = 0)))

    val stopped = model.solve(adapter(LpAdapterResult(LpStatus.Stopped)))
    try {
      assert(!stopped.candidate.available)
      assert(stopped.objectiveValue.isNaN)
    } finally stopped.close()
  }

  test("required duals cover every constraint") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("partial duals")
    val x = model.variable("x", lowerBound = -1.0, upperBound = Some(1.0))
    model += x
    model += (x >= -1.0).named("lower")
    model += (x <= 1.0).named("upper")
    val values = Some(model.candidateValues(Seq(x -> 0.0)))
    val oneDual = Some(sc.parallelize(Seq(model.inspect.constraints.first().id -> 0.0)))
    val adapter = new LpSolverAdapter {
      override val name = "partial-dual-fixture"
      override val capabilities = LpSolverCapabilities(duals = true)
      override def prepare(view: LpModelView, options: LpAdapterOptions): LpAdapterSession = new LpAdapterSession {
        override def solve(): LpAdapterResult = LpAdapterResult(LpStatus.Optimal, values, rowDuals = oneDual)
        override def close(): Unit = ()
      }
    }

    intercept[LpModelException](model.solve(adapter, LpAdapterOptions(requireDuals = true)))
  }

  test("CLI failure, missing executable, malformed output, timeout, cancellation and retained artifacts") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("lifecycle")
    model.variable("x", upperBound = Some(1.0))
    val root = Files.createTempDirectory("adapter lifecycle ")
    def cli(code: String, retain: Boolean = false, executable: String = "python3"): LpCommandAdapter =
      new LpCommandAdapter(LpCommandOptions(retainArtifacts = retain, temporaryRoot = Some(root))) {
        override val name = "failure-fixture"
        override val capabilities = LpSolverCapabilities()
        override protected def command(input: Path, output: Path, options: LpAdapterOptions): Seq[String] =
          Seq(executable, "-c", code, output.toString)
        override protected def parse(path: Path, mapping: LpExportMapping, options: LpAdapterOptions): LpAdapterResult =
          throw new IllegalArgumentException("malformed solution")
      }
    try {
      intercept[java.io.IOException](model.solve(cli("", executable = "/missing/spark-lp-executable")))
      assert(empty(root))

      val failure = intercept[IllegalStateException](model.solve(cli("import sys; print('diagnostic'); sys.exit(7)")))

      assert(failure.getMessage.contains("diagnostic"))
      assert(empty(root))
      intercept[IllegalStateException](model.solve(cli("pass")))
      intercept[IllegalArgumentException](model.solve(cli("import sys; open(sys.argv[1],'w').write('bad')")))
      assert(empty(root))

      val timeout = model.solve(cli("import time; time.sleep(10)"), LpAdapterOptions(timeLimit = Some(100.millis)))
      try {
        assert(timeout.status == LpStatus.Stopped)
        assert(!timeout.candidate.available)
      } finally timeout.close()

      val cancel = model.solve(cli("raise RuntimeError('must not run')"), LpAdapterOptions(shouldStop = () => true))
      try {
        assert(cancel.status == LpStatus.Stopped)
        assert(empty(root))
      } finally cancel.close()

      val retained = model.solve(cli("import time; time.sleep(10)", retain = true), LpAdapterOptions(timeLimit = Some(100.millis)))
      try {
        assert(Files.isDirectory(java.nio.file.Paths.get(retained.backend.get.diagnostics("artifacts"))))
      } finally retained.close()
    } finally {
      val paths = Files.walk(root)
      try paths.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.deleteIfExists) finally paths.close()
    }
  }
}
