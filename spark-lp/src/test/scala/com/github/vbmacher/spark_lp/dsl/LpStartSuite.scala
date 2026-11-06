package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpStartSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("full and partial LP starts preserve shifted, reflected, free and keyed coordinates") {
    implicit val ss: SparkSession = spark
    val local = spark
    import local.implicits._
    val model = LpProblem("coordinates", Maximize)
    val x = model.variable("shifted", -2.0, Some(3.0))
    val upper = model.variable("reflected", Double.NegativeInfinity, Some(2.0))
    val free = model.variable("free", Double.NegativeInfinity)
    val keyed = model.variables("keyed", Seq("a", "b").toDF("key"), $"key", upperBound = Some(2.0))
    model += x + upper + keyed("a") + 7.0
    model += (x + upper <= 3.0)
    model += (free === -1.0)
    model += (keyed.sum <= 2.0)
    val config = SolveConfig(newtonSolver = NewtonSolver.ConjugateGradient)

    val cold = model.solve(config)
    val previous = cold.asStart()
    cold.close()

    val full = model.solve(config.copy(start = Some(previous)))
    val partial = model.start(Seq(x -> -1.0, keyed("a") -> 1.0))

    val hinted = model.solve(config.copy(start = Some(partial)))
    try {
      assert(math.abs(full.objectiveValue - 12.0) < 1e-6)
      assert(math.abs(hinted.objectiveValue - 12.0) < 1e-6)
      assert(full.start.exists(s => s.used && s.complete && !s.seededIncumbent))
      assert(hinted.start.exists(s => s.used && !s.complete && !s.feasible))
      assert(math.abs(full.value(free) + 1.0) < 1e-6)
      assert(full.values(keyed).count() == 2)
      intercept[LpModelException](cold.asStart())
    } finally { full.close(); hinted.close(); previous.close(); partial.close() }
  }

  test("a validated MIP start seeds the incumbent before a cooperative stop") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("incumbent", Maximize)
    val x = model.variable("x", category = Binary)
    val y = model.variable("y", category = Binary)
    model += 2.0 * x + y + 5.0
    model += (2.0 * x + 2.0 * y <= 3.0)
    val start = model.start(Seq(x -> 1.0, y -> 0.0))
    val config = SolveConfig(start = Some(start), mip = MipConfig(control = MipControl(shouldStop = () => true)))

    val stopped = model.solve(config)
    val coldStop = model.solve(config.copy(start = None))
    val solved = model.solve(SolveConfig(start = Some(start)))
    val cold = model.solve()
    try {
      assert(stopped.status == LpStatus.Stopped)
      assert(stopped.objectiveValue == 7.0)
      assert(stopped.value(x) == 1.0)
      assert(stopped.candidate.feasible)
      assert(stopped.start.exists(s => s.used && s.seededIncumbent))
      assert(stopped.mip.exists(m => m.processedNodes == 0 && m.bestBound.isEmpty))
      assert(!coldStop.candidate.available)
      assert(solved.status == LpStatus.Optimal)
      assert(math.abs(solved.objectiveValue - cold.objectiveValue) < 1e-6)
    } finally { stopped.close(); coldStop.close(); solved.close(); cold.close(); start.close() }
  }

  test("malformed starts are rejected, bounds can be repaired and fractional hints do not create incumbents") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("start validation", Maximize)
    val x = model.variable("x", category = Binary)
    val y = model.variable("y", category = Binary)
    model += x + y
    model += (x + y <= 1.5)
    val id = model.inspect.variables.filter(_.name == "x").first().id
    val malformed = Seq(Seq(LpCandidateValue(id, Double.NaN)), Seq(LpCandidateValue(id, 0.0), LpCandidateValue(id, 1.0)),
      Seq(LpCandidateValue(LpVariableId(999, ""), 0.0)))
    val config = SolveConfig(mip = MipConfig(control = MipControl(shouldStop = () => true)))
    malformed.foreach { records =>
      val start = model.start(sc.parallelize(records))

      val result = model.solve(config.copy(start = Some(start)))
      try {
        assert(result.start.exists(s => s.disposition == LpStartDisposition.Rejected && !s.used) && !result.candidate.available)
      } finally { result.close(); start.close() }
    }
    val fractional = model.start(Seq(x -> 0.5, y -> 0.5))
    val repaired = model.start(Seq(x -> 2.0, y -> -1.0))
    val strict = model.start(model.candidateValues(Seq(x -> 0.5, y -> 0.5)), LpStartConfig(requireFeasible = true))
    val missing = model.start(model.candidateValues(Seq(x -> 1.0)), LpStartConfig(allowPartial = false))
    try {
      val hint = model.solve(config.copy(start = Some(fractional)))
      val fixed = model.solve(config.copy(start = Some(repaired)))
      val rejected = model.solve(config.copy(start = Some(strict)))
      val incomplete = model.solve(config.copy(start = Some(missing)))
      try {
        assert(!hint.candidate.available)
        assert(hint.start.exists(s => !s.feasible && !s.seededIncumbent))
        assert(fixed.candidate.feasible)
        assert(fixed.start.exists(_.disposition == LpStartDisposition.Repaired))
        assert(rejected.start.exists(_.disposition == LpStartDisposition.Rejected))
        assert(incomplete.start.exists(_.disposition == LpStartDisposition.Rejected))
      } finally { hint.close(); fixed.close(); rejected.close(); incomplete.close() }
    } finally { fractional.close(); repaired.close(); strict.close(); missing.close() }
  }

  test("source reuse revalidates model changes, closure and unsupported backends") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("reuse")
    val x = model.variable("x", upperBound = Some(2.0))
    model += x
    model += (x >= 1.0)

    val solution = model.solve()
    val start = solution.asStart(LpStartConfig(requireFeasible = true))
    solution.close()
    try {
      x.setBounds(1.5, Some(2.0))

      val result = model.solve(SolveConfig(start = Some(start)))
      try {
        assert(math.abs(result.value(x) - 1.5) < 1e-6)
        assert(result.start.exists(_.disposition == LpStartDisposition.Rejected))
      } finally result.close()

      intercept[LpModelException](LpProblem("different").solve(SolveConfig(start = Some(start))))
      intercept[LpModelException](model.solve(new HighsPythonAdapter, LpAdapterOptions(start = Some(start))))
    } finally start.close()

    intercept[LpModelException](model.solve(SolveConfig(start = Some(start))))
  }

  test("starts respect SOS, full presolve, rounding feasibility and analytical dispatch") {
    implicit val ss: SparkSession = spark
    val sos = LpProblem("SOS start")
    val a = sos.variable("a", upperBound = Some(2.0))
    val b = sos.variable("b", upperBound = Some(2.0))
    sos += a + b
    sos += (a + b >= 1.0)
    sos.addSos1("one", Seq(a -> 0.0, b -> 1.0))
    val start = sos.start(Seq(a -> 1.0, b -> 0.0))

    val stopped = sos.solve(SolveConfig(start = Some(start), presolve = PresolveConfig(effort = PresolveEffort.Full),
      mip = MipConfig(control = MipControl(shouldStop = () => true))))
    try {
      assert(stopped.status == LpStatus.Stopped)
      assert(stopped.candidate.feasible)
      assert(stopped.value(a) == 1.0)
      assert(stopped.start.exists(_.seededIncumbent))
    } finally { stopped.close(); start.close() }
    val rounding = LpProblem("rounding")
    val x = rounding.variable("x", category = Binary)
    val y = rounding.variable("y", upperBound = Some(2.0))
    rounding += (1e8 * x + y === 1e8 - 1.0)
    val hint = rounding.start(Seq(x -> (1.0 - 1e-8), y -> 0.0))

    val noIncumbent = rounding.solve(SolveConfig(start = Some(hint), mip = MipConfig(control = MipControl(shouldStop = () => true))))
    try {
      assert(!noIncumbent.candidate.available && noIncumbent.start.exists(s => !s.feasible && !s.seededIncumbent))
    } finally { noIncumbent.close(); hint.close() }
    val box = LpProblem("analytical")
    val z = box.variable("z", upperBound = Some(1.0))
    val ignored = box.start(Seq(z -> 0.5))

    val result = box.solve(SolveConfig(start = Some(ignored)))
    try {
      assert(result.start.exists(s => s.disposition == LpStartDisposition.Unsupported && !s.used))
    } finally { result.close(); ignored.close() }
  }
}
