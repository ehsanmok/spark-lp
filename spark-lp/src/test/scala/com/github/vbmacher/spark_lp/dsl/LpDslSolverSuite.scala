package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.compiler.LpCompiler

import com.github.vbmacher.spark_lp.LP
import com.github.vbmacher.spark_lp.TestingUtils._
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.funsuite.AnyFunSuite

/** Driver-side mutable source: lets a test change what a lazy RDD produces between two solves. */
object LpDslSolverSuiteState {
  @volatile var costs: Map[String, Double] = Map("a" -> 1.0, "b" -> 2.0)
}

class LpDslSolverSuite extends AnyFunSuite with DataFrameSuiteBase {

  test("Auto policy and explicit overrides remain independent of the historical 5000-row boundary") {
    assert(SolveConfig().resolvedNewtonSolver(10000) == NewtonSolver.Cholesky)
    assert(SolveConfig().resolvedNewtonSolver(10001) == NewtonSolver.ConjugateGradient)
    Seq(1000, 1001, 5000, 5001).foreach { m =>
      assert(SolveConfig().resolvedNewtonSolver(m) == NewtonSolver.Cholesky)
      assert(SolveConfig(newtonSolver = NewtonSolver.Cholesky).resolvedNewtonSolver(m) == NewtonSolver.Cholesky)
    }

    assert(SolveConfig(maxLocalConstraints = 50).resolvedNewtonSolver(51) == NewtonSolver.ConjugateGradient)
    assert(SolveConfig(newtonSolver = NewtonSolver.ConjugateGradient).resolvedNewtonSolver(1) == NewtonSolver.ConjugateGradient)
  }

  // fixture from LPSuite; optimum 12.083 at (1.667, 5.833, 40, 0, 0, 13.333, 9.167)
  private val cArray = Array(2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0)
  private val BArray = Array(
    Array(12.0, 16.0, 30.0, 1.0, 0.0),
    Array(24.0, 16.0, 12.0, 0.0, 1.0),
    Array(-1.0, 0.0, 0.0, 0.0, 0.0),
    Array(0.0, -1.0, 0.0, 0.0, 0.0),
    Array(0.0, 0.0, -1.0, 0.0, 0.0),
    Array(0.0, 0.0, 0.0, 1.0, 0.0),
    Array(0.0, 0.0, 0.0, 0.0, 1.0))
  private val bArray = Array(120.0, 120.0, 120.0, 15.0, 15.0)

  test("solveSummary reports convergence, iterations and residuals; solve keeps its contract") {
    implicit val ss: SparkSession = spark
    val c: RDD[DenseVector] = sc.parallelize(cArray, 2).glom.map(new DenseVector(_))
    val at = sc.parallelize(BArray.toSeq, 2).map(v => Vectors.dense(v))
    val b = new DenseVector(bArray)

    val summary = LP.solveSummary(c, at, b)

    assert(summary.termination == LP.Termination.Converged)
    assert(summary.iterations > 0)
    assert(summary.iterations <= 50)
    assert(summary.primalResidual < 1e-8)
    assert(summary.dualResidual < 1e-8)
    assert(summary.dualityGap < 1e-8)
    assert(summary.objectiveValue ~== 12.083 absTol 1e-3)

    val expected = Vectors.dense(1.66666667, 5.83333333, 40.0, 0.0, 0.0, 13.33333333, 9.16666667)
    val xx = Vectors.dense(summary.x.flatMap(_.toArray).collect())

    assert(xx ~== expected absTol 1e-6)

    val c2: RDD[DenseVector] = sc.parallelize(cArray, 2).glom.map(new DenseVector(_))
    val at2 = sc.parallelize(BArray.toSeq, 2).map(v => Vectors.dense(v))

    val (value, _) = LP.solve(c2, at2, b)

    assert(value ~== summary.objectiveValue absTol 1e-9)
  }

  test("graceful stop retains the completed iterate and leaves Spark usable") {
    implicit val ss: SparkSession = spark
    def run(stop: Option[Int => Boolean], limit: Int) = {
      val c: RDD[DenseVector] = sc.parallelize(cArray, 2).glom.map(new DenseVector(_))
      val at = sc.parallelize(BArray.toSeq, 2).map(v => Vectors.dense(v))
      LP.solveSummary(c, at, new DenseVector(bArray), tolerance = 1e-14,
        maxIter = limit, stopAfterIteration = stop)
    }
    val stopped = run(Some(_ >= 2), 50)
    val limited = run(None, 2)
    try {
      assert(stopped.termination == LP.Termination.Stopped)
      assert(stopped.iterations == 2)
      assert(stopped.objectiveValue ~== limited.objectiveValue absTol 1e-9)
      assert(Vectors.dense(stopped.x.flatMap(_.toArray).collect()) ~==
        Vectors.dense(limited.x.flatMap(_.toArray).collect()) absTol 1e-9)

      assert(!Thread.currentThread().isInterrupted)
      assert(spark.range(2).count() == 2)
    } finally {
      stopped.x.unpersist(blocking = false)
      limited.x.unpersist(blocking = false)
    }
  }

  test("IterationLimit is truthful: residuals returned for a non-converged iterate") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val ingredients = Seq(
      ("chicken", 0.013, 0.100, 0.080, 0.001, 0.002),
      ("beef", 0.008, 0.200, 0.100, 0.005, 0.005)
    ).toDF("name", "cost", "protein", "fat", "fibre", "salt")

    val model = LpProblem("limited", Minimize)
    val amount = model.variables("amount", ingredients, $"name")
    model += lpSum(amount * $"cost")
    model += (lpSum(amount) === 100.0).named("total_weight")
    model += (lpSum(amount * $"protein") >= 8.0).named("protein_min")
    model += (lpSum(amount * $"salt") <= 0.4).named("salt_max")

    val solution = model.solve(SolveConfig(maxIterations = 1))

    assert(solution.status == LpStatus.IterationLimit)
    assert(solution.iterations == 1)
    assert(!solution.residuals.primal.isNaN)
    assert(!solution.residuals.dual.isNaN)
    assert(!solution.residuals.gap.isNaN)
    // constraints diagnostics stay readable even for a non-feasible iterate

    assert(solution.constraints.count() == 3)
  }

  test("Auto switches to the matrix-free CG solver beyond maxLocalConstraints") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val domain = Seq("alpha", "bravo").toDF("k")
    val model = LpProblem("cg-auto", Minimize)
    val v = model.variables("v", domain, $"k")
    model += lpSum(v)
    model += (lpSum(v) >= 4.0).named("floor")

    // maxLocalConstraints = 0: every model exceeds the Cholesky budget, so Auto must go matrix-free

    val solution = model.solve(SolveConfig(maxLocalConstraints = 0))

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 4.0 absTol 1e-6)
  }

  test("explicit Cholesky still enforces maxLocalConstraints") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val domain = Seq("alpha", "bravo").toDF("k")
    val model = LpProblem("cho-budget", Minimize)
    val v = model.variables("v", domain, $"k")
    model += lpSum(v)
    model += (lpSum(v) >= 4.0).named("floor")

    val error = intercept[LpModelException](
      model.solve(SolveConfig(maxLocalConstraints = 0, newtonSolver = NewtonSolver.Cholesky)))

    assert(error.getMessage.contains("maxLocalConstraints"))
    assert(error.getMessage.contains("ConjugateGradient"))
  }

  test("linearly dependent constraint rows raise LpNumericalException naming the precondition") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("singular", Minimize)
    val x = model.variable("x")
    val y = model.variable("y")
    val z = model.variable("z")
    model += x + y + z
    model += (x + y === 1.0).named("r1")
    model += (y + z === 1.0).named("r2")
    // r3 = r1 + r2: not a duplicate row, but the Gramian is exactly singular
    model += (x + 2.0 * y + z === 2.0).named("r3")

    val error = intercept[LpNumericalException](model.solve())

    assert(error.phase == "initialization")
    assert(error.completedIterations == 0)
    assert(error.getMessage.contains("full row rank"))
  }

  test("compilation and solutions are deterministic across runs") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    // domain deliberately unsorted: ordering must come from the encoded key, not partition order
    val domain = Seq("delta", "alpha", "charlie", "bravo").toDF("k")

    def build() = {
      val model = LpProblem("det", Minimize)
      val v = model.variables("v", domain, $"k")
      model += lpSum(v)
      model += (lpSum(v) >= 4.0).named("floor")
      (model, v)
    }

    val (model1, _) = build()
    val (model2, _) = build()
    val compiled1 = new LpCompiler(model1, SolveConfig()).compile()
    val compiled2 = new LpCompiler(model2, SolveConfig()).compile()

    val layout1 = compiled1.sortedCols.collect().map { case (g, cd) => (g, cd.setIndex, cd.enc, cd.kind, cd.cost) }
    val layout2 = compiled2.sortedCols.collect().map { case (g, cd) => (g, cd.setIndex, cd.enc, cd.kind, cd.cost) }

    assert(layout1.toSeq == layout2.toSeq)
    assert(compiled1.b.toArray.toSeq == compiled2.b.toArray.toSeq)
    // user columns are sorted by encoded key: alpha, bravo, charlie, delta, then the slack

    assert(layout1.map(_._1).toSeq == (0L until compiled1.numCols).toSeq)

    val (model3, v3) = build()
    val (model4, v4) = build()

    val values3 = model3.solve().values(v3).select("k", "lp_value").collect()
      .map(r => r.getString(0) -> r.getDouble(1)).toMap

    val values4 = model4.solve().values(v4).select("k", "lp_value").collect()
      .map(r => r.getString(0) -> r.getDouble(1)).toMap

    assert(values3.keySet == values4.keySet)
    values3.keySet.foreach { k => assert(values3(k) ~== values4(k) absTol 1e-9) }
  }

  test("evaluate-on-solve: a changed source changes the next solve, not the current one") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    LpDslSolverSuiteState.costs = Map("a" -> 1.0, "b" -> 2.0)

    val keys = sc.parallelize(Seq("a", "b"), 2)
    val rowRdd = keys.map(k => Row(k, LpDslSolverSuiteState.costs(k)))
    val schema = StructType(Seq(StructField("k", StringType), StructField("cost", DoubleType)))
    val lazySource = spark.createDataFrame(rowRdd, schema)

    val model = LpProblem("lazy", Minimize)
    val v = model.variables("v", lazySource, $"k")
    model += lpSum(v * $"cost")
    model += (lpSum(v) === 1.0).named("mass")

    val first = model.solve()

    assert(first.objectiveValue ~== 1.0 absTol 1e-4, "with cost(a)=1 all mass goes to a")

    LpDslSolverSuiteState.costs = Map("a" -> 3.0, "b" -> 2.0)

    val second = model.solve()

    assert(second.objectiveValue ~== 2.0 absTol 1e-4, "the re-solve reads the changed source")
  }

  test("compilation never materialises a dense n x m structure: AT rows are sparse and aligned with c") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val ingredients = Seq(
      ("chicken", 0.013, 0.100, 0.002),
      ("beef", 0.008, 0.200, 0.005)
    ).toDF("name", "cost", "protein", "salt")

    val model = LpProblem("sparse", Minimize)
    val amount = model.variables("amount", ingredients, $"name")
    model += lpSum(amount * $"cost")
    model += (lpSum(amount) === 100.0).named("total_weight")
    model += (lpSum(amount * $"protein") >= 8.0).named("protein_min")
    model += (lpSum(amount * $"salt") <= 0.4).named("salt_max")

    val compiled = new LpCompiler(model, SolveConfig()).compile()

    assert(compiled.numRows == 3)
    assert(compiled.numCols == 4, "two user variables plus one slack per inequality")

    val atRows = compiled.AT.collect()

    assert(atRows.length == 4)
    atRows.foreach { row =>
      assert(row.isInstanceOf[SparseVector], "AT rows must be sparse vectors, never dense structures")
      assert(row.size == compiled.numRows)
    }
    // slack columns carry exactly one entry; nothing is padded to dense

    assert(atRows.map(_.numNonzeros).sum == 2 * 3 + 2)

    // the solver's partitioning contract: partition i of c has as many elements as AT has rows there,
    // and no partition is empty
    val cSizes = compiled.c.map(_.size).collect()
    val atCounts = compiled.AT.mapPartitions(it => Iterator.single(it.size), preservesPartitioning = true).collect()

    assert(cSizes.toSeq == atCounts.toSeq)
    assert(cSizes.forall(_ > 0), "the compiler must not produce empty partitions")
  }
}
