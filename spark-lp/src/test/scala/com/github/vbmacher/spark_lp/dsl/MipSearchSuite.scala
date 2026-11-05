package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class MipSearchSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("cuts and strong branching preserve the known knapsack optimum") {
    implicit val ss: SparkSession = spark
    val modes = Vector(MipSearchConfig(), MipSearchConfig(cuts = MipCutsConfig(enabled = true)),
      MipSearchConfig(strongBranching = StrongBranchingConfig(enabled = true, maxCandidates = 2, maxIterations = 8, maxProbes = 12)),
      MipSearchConfig(cuts = MipCutsConfig(enabled = true),
        strongBranching = StrongBranchingConfig(enabled = true, maxCandidates = 2, maxIterations = 8, maxProbes = 12)))
    modes.foreach { policy =>
      val model = LpProblem("search", Maximize)
      val weights = Vector(2.0, 3.0, 4.0, 5.0)
      val profits = Vector(4.0, 5.0, 7.0, 8.0)
      val xs = Vector.tabulate(weights.size)(index => model.variable(s"item$index", category = Binary))
      model += lpDot(profits, xs) + 7.0
      model += (lpDot(weights, xs) <= 7.0)
      val expected = 19.0

      val result = model.solve(SolveConfig(mip = MipConfig(search = policy)))
      try {
        assert(result.status == LpStatus.Optimal)
        assert(math.abs(result.objectiveValue - expected) < 1e-6)

        val validation = model.validateCandidate(model.candidateValues(xs.map(v => v -> result.value(v))), CandidateValidationConfig(tolerance = 1e-6))
        try {
          assert(validation.feasible)
        } finally validation.close()
        val statistics = result.mip.get.search
        if (policy.cuts.enabled) assert(statistics.globalCuts > 0 && statistics.globalCuts + statistics.localCuts <= policy.cuts.maxCuts)
        if (policy.strongBranching.enabled) assert(statistics.strongProbes > 0 && statistics.strongProbes <= policy.strongBranching.maxProbes)
      } finally result.close()
    }
  }

  test("incomplete bounded probes preserve a valid bound and do not create an incumbent") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("search", Maximize)
    val weights = Vector(2.0, 3.0, 4.0, 5.0)
    val profits = Vector(4.0, 5.0, 7.0, 8.0)
    val xs = Vector.tabulate(weights.size)(index => model.variable(s"item$index", category = Binary))
    model += lpDot(profits, xs) + 7.0
    model += (lpDot(weights, xs) <= 7.0)
    val optimum = 19.0

    val result = model.solve(SolveConfig(mip = MipConfig(maxNodes = 1,
      search = MipSearchConfig(strongBranching = StrongBranchingConfig(enabled = true, maxIterations = 1, maxProbes = 2)))))
    try {
      assert(result.status == LpStatus.IterationLimit)
      assert(!result.candidate.available)
      assert(result.mip.get.bestBound.exists(_ >= optimum - 1e-6))
      assert(result.mip.get.search.strongProbes == 2)
      assert(result.mip.get.processedNodes == 1)
    } finally result.close()
  }

  test("parallel nodes coordinate bounds and incumbents while respecting node and memory budgets") {
    implicit val ss: SparkSession = spark
    val coordinator = Thread.currentThread()
    val model = LpProblem("parallel", Maximize)
    val xs = Vector.tabulate(4)(i => model.variable(s"x$i", category = Binary))
    model += lpDot(Vector(1.0, 2.0, 3.0, 4.0), xs)
    model += (lpDot(Vector.fill(4)(2.0), xs) <= 5.0)
    val events = scala.collection.mutable.ArrayBuffer.empty[MipProgress]
    val policy = MipSearchConfig(parallelNodes = 2)

    val result = model.solve(SolveConfig(mip = MipConfig(search = policy, control = MipControl(
      onProgress = event => { assert(Thread.currentThread() eq coordinator); events += event },
      onNodeProgress = (_, _) => assert(Thread.currentThread() eq coordinator)))))
    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.objectiveValue - 7.0) < 1e-6)
      assert(result.mip.get.search.peakConcurrentNodes == 2)
      assert(events.flatMap(_.bestBound).forall(_ >= 7.0 - 1e-6))
      assert(events.flatMap(_.incumbent).forall(_ <= 7.0 + 1e-6))
      assert(result.mip.get.search.estimatedPeakLocalBytes <= policy.maxConcurrentLocalBytes)
    } finally result.close()

    val limited = model.solve(SolveConfig(mip = MipConfig(maxNodes = 2, search = policy)))
    try {
      assert(limited.status == LpStatus.IterationLimit)
      assert(limited.mip.get.processedNodes <= 2)
    } finally limited.close()

    intercept[LpModelException](model.solve(SolveConfig(mip = MipConfig(search = policy.copy(maxConcurrentLocalBytes = 1)))))
    import scala.collection.JavaConverters._

    assert(!Thread.getAllStackTraces.keySet().asScala.exists(t => t.isAlive && t.getName.startsWith("spark-lp-mip-")))
  }

  test("parallel cancellation retains a start and callback failures clean up worker resources") {
    implicit val ss: SparkSession = spark
    val before = sc.getPersistentRDDs.keySet
    val model = LpProblem("parallel stop", Maximize)
    val xs = Vector.tabulate(4)(i => model.variable(s"x$i", category = Binary))
    model += lpDot(Vector(1.0, 2.0, 3.0, 4.0), xs)
    model += (lpDot(Vector.fill(4)(2.0), xs) <= 5.0)
    val start = model.start(xs.zip(Vector(0.0, 0.0, 1.0, 1.0)))
    val stop = new java.util.concurrent.atomic.AtomicBoolean(false)
    val policy = MipSearchConfig(parallelNodes = 2)

    val result = model.solve(SolveConfig(start = Some(start), mip = MipConfig(search = policy,
      control = MipControl(shouldStop = () => stop.get(), onNodeProgress = (number, _) => if (number >= 2) stop.set(true)))))
    try {
      assert(result.status == LpStatus.Stopped)
      assert(result.candidate.feasible)
      assert(result.objectiveValue == 7.0)
      assert(result.start.exists(_.seededIncumbent))
      assert(result.mip.get.bestBound.exists(_ >= 7.0 - 1e-6))
    } finally { result.close(); start.close() }

    intercept[IllegalStateException](model.solve(SolveConfig(mip = MipConfig(search = policy,
      control = MipControl(onNodeProgress = (number, _) => if (number >= 2) throw new IllegalStateException("callback failure"))))))
    import scala.collection.JavaConverters._

    assert(!Thread.getAllStackTraces.keySet().asScala.exists(t => t.isAlive && t.getName.startsWith("spark-lp-mip-")))
    assert(sc.getPersistentRDDs.keySet.subsetOf(before))
  }

  test("combined search preserves shifted integer minimization and SOS semantics") {
    implicit val ss: SparkSession = spark
    val policy = MipSearchConfig(cuts = MipCutsConfig(enabled = true, maxRounds = 1, maxCuts = 8),
      strongBranching = StrongBranchingConfig(enabled = true, maxCandidates = 2, maxIterations = 8, maxProbes = 8),
      parallelNodes = 2)
    val integers = LpProblem("shifted minimization", Minimize)
    val x = integers.variable("x", lowerBound = -1.0, upperBound = Some(2.0), category = Integer)
    val y = integers.variable("y", lowerBound = -1.0, upperBound = Some(2.0), category = Integer)
    integers += 3.0 * x + 4.0 * y + 7.0
    integers += (2.0 * x + 3.0 * y >= 2.0)
    val expected = (for (a <- -1 to 2; b <- -1 to 2 if 2 * a + 3 * b >= 2) yield 3 * a + 4 * b + 7).min

    val result = integers.solve(SolveConfig(mip = MipConfig(search = policy)))
    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.objectiveValue - expected) < 1e-6)

      val check = integers.validateCandidate(integers.candidateValues(Seq(x -> result.value(x), y -> result.value(y))),
        CandidateValidationConfig(tolerance = 1e-6))
      try {
        assert(check.feasible)
      } finally check.close()

      assert(result.mip.get.bestBound.exists(_ <= expected + 1e-6))
    } finally result.close()

    val sos = LpProblem("parallel SOS", Maximize)
    val members = Vector.tabulate(3)(i => sos.variable(s"s$i", upperBound = Some(1.0)))
    sos += lpDot(Vector(1.0, 2.0, 3.0), members)
    sos.addSos1("selection", members.zip(Vector(1.0, 2.0, 3.0)))

    val selected = sos.solve(SolveConfig(mip = MipConfig(search = policy)))
    try {
      assert(selected.status == LpStatus.Optimal)
      assert(math.abs(selected.objectiveValue - 3.0) < 1e-6)

      val check = sos.validateCandidate(sos.candidateValues(members.map(v => v -> selected.value(v))),
        CandidateValidationConfig(tolerance = 1e-6))
      try {
        assert(check.feasible)
      } finally check.close()
    } finally selected.close()
  }


  test("parallel deadlines before and during root initialization preserve candidate availability") {
    implicit val ss: SparkSession = spark
    import scala.concurrent.duration._
    for (inside <- Vector(false, true)) {
      val model = LpProblem("search", Maximize)
      val weights = Vector(2.0, 3.0, 4.0, 5.0)
      val profits = Vector(4.0, 5.0, 7.0, 8.0)
      val xs = Vector.tabulate(weights.size)(index => model.variable(s"item$index", category = Binary))
      model += lpDot(profits, xs) + 7.0
      model += (lpDot(weights, xs) <= 7.0)
      var clock = 0L
      val policy = MipSearchConfig(parallelNodes = 2, cuts = MipCutsConfig(enabled = true),
        strongBranching = StrongBranchingConfig(enabled = true))
      val control = MipControl(timeLimit = Some(1.second), nanoTime = () => clock,
        onProgress = _ => if (!inside) clock = 2000000000L,
        onNodeProgress = (_, _) => if (inside) clock = 2000000000L)

      val result = model.solve(SolveConfig(mip = MipConfig(search = policy, control = control)))
      try {
        assert(result.status == LpStatus.Stopped)
        assert(result.stopReason.contains(com.github.vbmacher.spark_lp.StopReason.TimeLimit))
        assert(!result.candidate.available)
        assert(result.mip.get.incumbent.isEmpty)
      } finally result.close()
    }
    import scala.collection.JavaConverters._

    assert(!Thread.getAllStackTraces.keySet().asScala.exists(t => t.isAlive && t.getName.startsWith("spark-lp-mip-")))
  }


  test("parallel memory preflight includes uncut Cholesky when cuts are enabled") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("uncut memory", Minimize)
    val xs = Vector.tabulate(4)(i => model.variable(s"x$i", category = Binary))
    model += lpDot(Vector(1.0, 2.0, 3.0, 4.0), xs)
    model += (lpDot(Vector.fill(4)(2.0), xs) >= 3.0)
    // Five uncut rows need 1040 estimated bytes per Cholesky solve. Six cut rows
    // with a zero CG preconditioner budget need only 768; both paths must be budgeted.
    val policy = MipSearchConfig(parallelNodes = 2, maxConcurrentLocalBytes = 1800L,
      cuts = MipCutsConfig(enabled = true, maxCutsPerNode = 1, maxCuts = 1))

    intercept[LpModelException] {
      val result = model.solve(SolveConfig(mip = MipConfig(search = policy),
        cgConfig = com.github.vbmacher.spark_lp.newton.CgConfig(preconditionerMemoryBytes = 0L)))
      result.close()
    }
  }

}
