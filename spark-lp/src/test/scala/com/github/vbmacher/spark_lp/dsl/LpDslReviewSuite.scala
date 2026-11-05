package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpDslReviewSuite extends AnyFunSuite with DataFrameSuiteBase {

  test("Auto uses the shared crossover and the DSL can lower the Cholesky resource cap") {
    val limit = NewtonSolver.AutoCholeskyLimit.toLong

    assert(SolveConfig().resolvedNewtonSolver(10000) == NewtonSolver.Cholesky)
    assert(SolveConfig().resolvedNewtonSolver(10001) == NewtonSolver.ConjugateGradient)
    assert(SolveConfig().resolvedNewtonSolver(limit) == NewtonSolver.Cholesky)
    assert(SolveConfig().resolvedNewtonSolver(limit + 1) == NewtonSolver.ConjugateGradient)
    assert(SolveConfig(maxLocalConstraints = 0).resolvedNewtonSolver(1) == NewtonSolver.ConjugateGradient)
    assert(SolveConfig(maxLocalConstraints = 5000).resolvedNewtonSolver(5000) == NewtonSolver.Cholesky)
    assert(SolveConfig(maxLocalConstraints = 5000).resolvedNewtonSolver(5001) == NewtonSolver.ConjugateGradient)
    assert(SolveConfig(maxLocalConstraints = limit + 1).resolvedNewtonSolver(limit + 1) ==
      NewtonSolver.ConjugateGradient)

    assert(SolveConfig(newtonSolver = NewtonSolver.Cholesky).resolvedNewtonSolver(limit + 1) ==
      NewtonSolver.Cholesky)
  }

  test("composite keys cannot collide through embedded separators or escapes") {
    val keys = Seq(Seq("a", "b"), Seq("a\u001Fs:b"), Seq("a\\u001Fs:b"),
      Seq("a\u001Fs:b", "c"), Seq("a", "b\u001Fs:c"))

    assert(keys.map(KeyCodec.encodeParts).distinct.size == keys.size)
  }

  test("permuted equality coefficients are neither merged nor rejected as duplicates") {
    implicit val ss: SparkSession = spark
    for ((rhs, optimum) <- Seq(6.0 -> 2.4, 7.0 -> 2.6)) {
      val model = LpProblem("permuted")
      val x = model.variable("x")
      val y = model.variable("y")
      val z = model.variable("z")
      model += x + y + z
      model += (x + 2.0 * y + 3.0 * z === 6.0).named("first")
      model += (x + 3.0 * y + 2.0 * z === rhs).named("second")

      val result = model.solve()

      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.objectiveValue - optimum) < 1e-6)
      assert(result.constraints.collect().forall(r => r.isNullAt(7) && math.abs(r.getDouble(5)) < 1e-6))
    }
  }

  test("zero cost and zero RHS models start strictly inside the nonnegative cone") {
    implicit val ss: SparkSession = spark
    for (solver <- Seq(NewtonSolver.Cholesky, NewtonSolver.ConjugateGradient);
         rhs <- Seq(0.0, 1.0); cost <- Seq(0.0, 1.0)) {
      val model = LpProblem("zero")
      val x = model.variable("x")
      val y = model.variable("y")
      model += cost * x + 2.0 * cost * y
      model += (x + y === rhs)

      val result = model.solve(SolveConfig(newtonSolver = solver))

      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.objectiveValue - cost * rhs) < 1e-6)
      assert(math.abs(result.value(x) + result.value(y) - rhs) < 1e-6)
    }
  }

  test("non-finite objective constants are rejected before solving") {
    implicit val ss: SparkSession = spark
    for (constant <- Seq(Double.NaN, Double.PositiveInfinity)) {
      val model = LpProblem("constant")
      val x = model.variable("x")
      model += x + constant
      model += (x >= 1.0)

      assert(intercept[LpModelException](model.solve()).getMessage.contains("objective"))
    }
  }

  test("solves and validation failures release their temporary cached RDDs") {
    implicit val ss: SparkSession = spark
    val before = sc.getPersistentRDDs.keySet
    val model = LpProblem("cache")
    val x = model.variable("x")
    model += lpSum(x)
    model += (x >= 1.0)
    for (_ <- 1 to 2) {
      val result = model.solve()

      assert(math.abs(result.value(x) - 1.0) < 1e-6)
      result.close()

      assert(sc.getPersistentRDDs.keySet == before)
    }
    model += (x <= Double.NaN)

    intercept[LpModelException](model.solve())
    assert(sc.getPersistentRDDs.keySet == before)
  }

  test("sum and sumBy express a grouped allocation without relational plumbing") {
    implicit val ss: SparkSession = spark
    val session = spark
    import session.implicits._
    val domain = Seq(("a", "east", 1.0), ("b", "east", 3.0), ("c", "west", 2.0))
      .toDF("id", "region", "cost")
    val demand = Seq(("east", 4.0), ("west", 2.0)).toDF("region", "rhs")
    val model = LpProblem("allocation")
    val amount = model.variables("amount", domain, $"id")
    model += amount.sum($"cost")
    model += (amount.sumBy("region")() === demand).named("demand")
    model += (amount.sumBy("region")($"cost") <= 10.0).named("budget")

    val solution = model.solve()
    try {
      assert(solution.status == LpStatus.Optimal)
      assert(math.abs(solution.objectiveValue - 8.0) < 1e-6)
      assert(solution.constraints.count() == 4)
      val values = solution.values(amount).select("id", "lp_value").collect()
        .map(r => r.getString(0) -> r.getDouble(1)).toMap

      assert(math.abs(values("a") - 4.0) < 1e-6)
      assert(math.abs(values("b")) < 1e-6)
      assert(math.abs(values("c") - 2.0) < 1e-6)
    } finally solution.close()
  }

  test("bound-only and fully fixed models need no artificial constraints") {
    implicit val ss: SparkSession = spark
    for (lower <- Seq(0.0, 3.0)) {
      val model = LpProblem("bounds", Maximize)
      val x = model.variable("x", lowerBound = lower, upperBound = Some(3.0))
      model += 2.0 * x + 1.0

      val solution = model.solve()
      try {
        assert(solution.status == LpStatus.Optimal)
        assert(math.abs(solution.value(x) - 3.0) < 1e-6)
        assert(math.abs(solution.objectiveValue - 7.0) < 1e-6)
      } finally solution.close()
    }
  }

  test("integer rounding never expands explicitly declared bounds") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("integer bounds")
    val x = model.variable("x", lowerBound = 0.0000001, upperBound = Some(1.0), category = Integer)
    model += lpSum(x)

    val solution = model.solve()
    try {
      assert(solution.status == LpStatus.Optimal)
      assert(solution.value(x) == 1.0)
    } finally solution.close()
  }

  test("MIP node limits do not claim optimality for a fractional relaxation") {
    implicit val ss: SparkSession = spark
    val before = sc.getPersistentRDDs.keySet
    val model = LpProblem("node limit", Maximize)
    val x = model.variable("x", upperBound = Some(2.0), category = Integer)
    model += lpSum(x)
    model += (x <= 1.5)

    val solution = model.solve(SolveConfig(mip = MipConfig(maxNodes = 1)))
    try {
      assert(solution.status == LpStatus.IterationLimit)
      assert(solution.objectiveValue.isNaN)
      assert(!solution.candidate.available)
      intercept[LpModelException](solution.value(x))
    } finally solution.close()

    assert(sc.getPersistentRDDs.keySet == before)
  }

  test("invalid numerical settings fail before submitting Spark jobs") {
    intercept[IllegalArgumentException](SolveConfig(tolerance = Double.NaN))
    intercept[IllegalArgumentException](SolveConfig(cgTolerance = Double.PositiveInfinity))
    intercept[IllegalArgumentException](SolveConfig(maxIterations = 0))
    intercept[IllegalArgumentException](SolveConfig(etaIteration = 1.0))
    intercept[IllegalArgumentException](MipConfig(maxNodes = 0))
    intercept[IllegalArgumentException](MipConfig(gapTolerance = Double.NaN))
  }

  test("dense elementwise operations reject unequal vector sizes") {
    import com.github.vbmacher.spark_lp.vectors.dense_vector.implicits._
    import org.apache.spark.mllib.linalg.Vectors
    val a = Vectors.dense(1.0, 2.0).toDense
    val b = Vectors.dense(-1.0).toDense

    intercept[IllegalArgumentException](a.entrywiseProd(b))
    intercept[IllegalArgumentException](a.entrywiseNegDiv(b))
  }

  test("the corrector uses the current complementarity on the second iteration") {
    import com.github.vbmacher.spark_lp.LP
    import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
    implicit val ss: SparkSession = spark
    val c = sc.parallelize(Seq(new DenseVector(Array(2.0, 4.0, -1.0, 3.0))), 1)
    val at = sc.parallelize(Seq(Vectors.dense(1.0, 3.0), Vectors.dense(2.0, 1.0),
      Vectors.dense(3.0, 4.0), Vectors.dense(4.0, 2.0)), 1)

    val summary = LP.solveSummary(c, at, Vectors.dense(10.0, 10.0).toDense, maxIter = 2)
    try {
      // Independent dense Newton calculation; using the initial mu gives 1.07188416266.

      assert(math.abs(summary.objectiveValue - 1.0716595749289968) < 1e-8)
      assert(summary.termination == LP.Termination.IterationLimit)
    } finally summary.x.unpersist(blocking = false)
  }

  test("numerical failure releases owned caches and preserves caller persistence") {
    import com.github.vbmacher.spark_lp.LP
    import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
    import org.apache.spark.storage.StorageLevel
    implicit val ss: SparkSession = spark
    val at = sc.parallelize(Seq(Vectors.dense(1.0, 1.0), Vectors.dense(1.0, 1.0)), 1)
      .persist(StorageLevel.DISK_ONLY)
    at.count()
    val before = sc.getPersistentRDDs.keySet
    val c = sc.parallelize(Seq(new DenseVector(Array(1.0, 2.0))), 1)
    try {
      intercept[LpNumericalException](LP.solveSummary(c, at, Vectors.dense(1.0, 1.0).toDense))
      assert(sc.getPersistentRDDs.keySet == before)
      assert(at.getStorageLevel == StorageLevel.DISK_ONLY)
    } finally at.unpersist(blocking = false)
  }
}
