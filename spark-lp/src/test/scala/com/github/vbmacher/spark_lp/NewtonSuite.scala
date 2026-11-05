package com.github.vbmacher.spark_lp

import com.github.vbmacher.spark_lp.newton.{CgConfig, NewtonSolver}

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, diag, norm}
import com.github.vbmacher.spark_lp.vectors.{DMatrix, DVector}
import com.github.vbmacher.spark_lp.vectors.dmatrix.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class NewtonSuite extends AnyFunSuite with DataFrameSuiteBase {
  private def vector(a: Array[Double]): DVector = sc.parallelize(a.toSeq, 2).glom.map(new DenseVector(_))
  private def local(v: DVector): BDV[Double] = new BDV(v.flatMap(_.values).collect())
  private def matrix(b: BDM[Double]): DMatrix =
    sc.parallelize((0 until b.rows).map(i => Vectors.dense((0 until b.cols).map(j => b(i, j)).toArray)), 2)

  test("Cholesky block boundaries preserve tiny and nonadjacent couplings") {
    def packed(g: BDM[Double]): Array[Double] =
      (0 until g.cols).flatMap(j => (0 to j).map(i => g(i, j))).toArray
    val g = BDM.eye[Double](5)
    g(0, 1) = 0.1
    g(3, 4) = 0.1

    assert(newton.CholeskyFactory.blockEnds(packed(g), 5).sameElements(Array(2, 3, 5)))
    g(1, 3) = 1e-100

    assert(newton.CholeskyFactory.blockEnds(packed(g), 5).sameElements(Array(5)))
    g(1, 3) = 0.0
    g(0, 4) = 0.1

    assert(newton.CholeskyFactory.blockEnds(packed(g), 5).sameElements(Array(5)))
  }

  test("block Cholesky solves weighted independent systems and preserves both right-hand sides") {
    val b = BDM((1.0, 2.0, 0.0, 0.0, 0.0), (2.0, -1.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 3.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0, 2.0),
      (0.0, 0.0, 0.0, 2.0, -1.0))
    val w = BDV(0.5, 2.0, 3.0, 0.7, 1.2)
    val weights = newton.Weights(vector(w.toArray.map(math.sqrt)), vector(w.toArray))
    val system = new newton.CholeskyFactory().build(matrix(b), 5, Some(weights))
    try {
      Seq(BDV(1.0, 2.0, 3.0, 4.0, 5.0), BDV(-3.0, 0.0, 1.0, -1.0, 2.0)).foreach { rhs =>
        val input = new DenseVector(rhs.toArray)

        val actual = new BDV(system.solve(input).values)

        assert(norm(actual - ((b.t * diag(w) * b) \ rhs)) < 1e-12)
        assert(input.values.sameElements(rhs.toArray))
      }
    } finally system.release()
  }

  test("matrix-free configuration rejects invalid regularization and memory settings") {
    intercept[IllegalArgumentException](CgConfig(primalRegularization = 0.0))
    intercept[IllegalArgumentException](CgConfig(dualRegularization = Double.NaN))
    intercept[IllegalArgumentException](CgConfig(primalRegularization = Double.PositiveInfinity))
    intercept[IllegalArgumentException](CgConfig(preconditionerRank = -1))
    intercept[IllegalArgumentException](CgConfig(preconditionerMemoryBytes = -1L))
  }

  test("partial Cholesky selects the updated Schur diagonal and applies the explicit preconditioner") {
    // Original diagonals select 0,1; after eliminating 0, row 2 must precede row 1.
    val g = BDM((10.0, 9.0, 0.0), (9.0, 9.0, 0.2), (0.0, 0.2, 8.0))
    val fetched = scala.collection.mutable.ArrayBuffer.empty[Int]
    val partial = new newton.PartialCholesky(diag(g).toArray, 2, j => {
      fetched += j
      (0 until 3).map(i => g(i, j)).toArray
    }, 1e-8)

    assert(partial.indices.sameElements(Array(0, 2)))
    assert(fetched.toSeq == Seq(0, 2))
    val rhs = BDV(1.0, 2.0, -3.0)
    // Only one row remains: its Schur diagonal is exact, so M = G.

    assert(norm(partial(rhs) - (g \ rhs)) < 1e-12)

    val rankOne = new newton.PartialCholesky(diag(g).toArray, 1,
      j => (0 until 3).map(i => g(i, j)).toArray, 1e-8)
    val explicit = g.copy
    explicit(1, 2) = 0.0
    explicit(2, 1) = 0.0

    assert(norm(rankOne(rhs) - (explicit \ rhs)) < 1e-12)
  }

  test("unstable pivots stop at the floor without fetching a dense Gramian") {
    val partial = new newton.PartialCholesky(Array(1e-8, 1e-8), 2,
      _ => fail("No column should be fetched for a pivot at the regularization floor"), 1e-8)

    assert(partial.indices.isEmpty)
    assert(norm(partial(BDV(1e-8, 2e-8)) - BDV(1.0, 2.0)) < 1e-12)
    partial.extendTo(2)

    assert(partial.indices.isEmpty)
    intercept[IllegalStateException] {
      new newton.PartialCholesky(Array(1.0, 1.0), 2, j =>
        if (j == 0) Array(1.0, 2.0) else Array(2.0, 1.0), 1e-8)
    }
  }

  test("incremental partial Cholesky matches fresh factors without refetching old pivots") {
    val g = BDM((10.0, 9.0, 0.0), (9.0, 9.0, 0.2), (0.0, 0.2, 8.0))
    val fetched = scala.collection.mutable.ArrayBuffer.empty[Int]
    val progress = scala.collection.mutable.ArrayBuffer.empty[Int]
    def column(j: Int): Array[Double] = (0 until 3).map(i => g(i, j)).toArray
    val incremental = new newton.PartialCholesky(diag(g).toArray, 0, j => {
      fetched += j
      column(j)
    }, 1e-8, completed => { progress += completed; () })
    val rhs = BDV(1.0, 2.0, 3.0)
    Seq(0, 1, 2, 3, 3, 1).foreach { rank =>
      incremental.extendTo(rank)
      val fresh = new newton.PartialCholesky(diag(g).toArray, fetched.size, column, 1e-8)

      assert(incremental.indices.sameElements(fresh.indices))
      assert(norm(incremental(rhs) - fresh(rhs)) < 1e-14)
    }

    assert(fetched.toSeq == Seq(0, 2, 1))
    assert(progress.toSeq == Seq(1, 2, 3))
    assert(norm(incremental(rhs) - (g \ rhs)) < 1e-12)
  }

  test("CG rank escalation reuses pivots within a system and rebuilds for new weights") {
    implicit val session: SparkSession = spark
    val m = 64
    val rows: DMatrix = sc.parallelize(0 to m, 4).map { i =>
      val entries = if (i == 0) Seq(0 -> 1.0)
        else if (i == m) Seq((m - 1) -> -1.0)
        else Seq((i - 1) -> -1.0, i -> 1.0)
      Vectors.sparse(m, entries)
    }.cache()
    val pivots = scala.collection.mutable.ArrayBuffer.empty[Int]
    val monitor = new SolveMonitor(SolveControl(onProgress = { e =>
      if (e.phase == SolvePhase.SystemSetup) e.work.flatMap(_.preconditionerRank).foreach(pivots += _)
    }))
    val factory = new newton.CgFactory(1e-10, 1, monitor = monitor)
    val rhs = new DenseVector(Array.tabulate(m)(i => (i % 7 + 1).toDouble))
    try {
      Seq(None, Some(newton.Weights(
        rows.mapPartitions(it => Iterator.single(new DenseVector(it.map(_ => math.sqrt(2.0)).toArray))),
        rows.mapPartitions(it => Iterator.single(new DenseVector(it.map(_ => 2.0).toArray)))))).foreach { weights =>
        pivots.clear()
        val system = factory.build(rows, m, weights)
        try {
          val solution = system.solve(rhs)

          assert(pivots.toSeq == (1 to m))
          val p = sc.broadcast(solution)
          try {
            val scale = if (weights.isDefined) 1.0 else 1.0 / (1.0 + factory.primalRegularization)
            val actual = newton.CgFactory.regularizedProduct(new DMatrixOps(rows), weights.map(_.squared),
              p, factory.dualRegularization, scale)

            assert(norm(new BDV(actual.values) - new BDV(rhs.values)) / norm(new BDV(rhs.values)) < 1e-10)
          } finally p.destroy()
        } finally system.release()
      }
    } finally rows.unpersist()
  }

  test("regularized products and both recovered directions match explicit augmented equations") {
    implicit val session: SparkSession = spark
    val b = BDM((1.0, 2.0), (2.0, -1.0), (0.0, 1.0))
    val rows = matrix(b)
    val x = BDV(2.0, 3.0, 4.0)
    val s = BDV(0.5, 2.0, 1.0)
    val rc = BDV(0.1, -0.2, 0.3)
    val rb = BDV(-0.4, 0.2)
    val rho = 0.2
    val delta = 0.3
    val weights = LP.regularizedWeights(vector(x.toArray), vector(s.toArray), rho)
    val w = diag(BDV.tabulate(3)(i => 1.0 / (s(i) / x(i) + rho)))
    val g = b.t * w * b + delta * BDM.eye[Double](2)
    val p = sc.broadcast(new DenseVector(Array(0.7, -0.3)))
    try {
      val actual = newton.CgFactory.regularizedProduct(new DMatrixOps(rows), Some(weights.squared), p, delta)

      assert(norm(new BDV(actual.values) - g * new BDV(p.value.values)) < 1e-12)
    } finally p.destroy()
    val factory = new newton.CgFactory(1e-12, 50, config = CgConfig(rho, delta))
    val system = factory.build(rows, 2, Some(weights))
    try {
      Seq(-x *:* s, BDV(-0.3, 0.2, -0.5)).foreach { q =>
        val h = rc + q /:/ x
        val rhs = -rb - b.t * w * h

        val dy = new BDV(system.solve(new DenseVector(rhs.toArray)).values)
        val (dxv, dsv) = weights.recoverDirections(vector(h.toArray), vector(rc.toArray),
          vector((b * dy).toArray), rho)
        val dx = local(dxv)
        val ds = local(dsv)

        assert(norm(dy - (g \ rhs)) < 1e-11)
        assert(norm(b.t * dx + delta * dy + rb) < 1e-11)
        assert(norm(b * dy + ds - rho * dx + rc) < 1e-11)
        assert(norm((s *:* dx) + (x *:* ds) - q) < 1e-11)
      }
    } finally system.release()
  }

  test("regularized initialization matches explicit least squares in both spaces") {
    val b = BDM((1.0, 2.0), (2.0, -1.0), (0.0, 1.0))
    val c = BDV(1.0, 2.0, 3.0)
    val rhs = BDV(2.0, 1.0)
    val rho = 0.2
    val delta = 0.3
    val g = (b.t * b) / (1.0 + rho) + delta * BDM.eye[Double](2)
    val y = g \ ((b.t * c) / (1.0 + rho))
    val xt = (b * (g \ rhs)) / (1.0 + rho)
    val st = c - b * y
    val xh = xt + math.max(0.0, -1.5 * xt.toArray.min)
    val sh = st + math.max(0.0, -1.5 * st.toArray.min)
    val comp = xh dot sh
    val expectedX = xh + (0.5 * comp / sh.toArray.sum)
    val expectedS = sh + (0.5 * comp / xh.toArray.sum)
    val result = Initialize.init(vector(c.toArray), matrix(b), new DenseVector(rhs.toArray),
      new newton.CgFactory(1e-12, 50, config = CgConfig(rho, delta))(spark))

    assert(norm(new BDV(result.lambda.values) - y) < 1e-10)
    assert(norm(local(result.x) - expectedX) < 1e-10)
    assert(norm(local(result.s) - expectedS) < 1e-10)
  }

  test("explicit rank obeys memory budget and failed inner solves cannot accept a loose defect") {
    val b = BDM((1.0, 2.0), (2.0, -0.5), (0.0, 1.0))
    val factory = new newton.CgFactory(1e-14, 1, config =
      CgConfig(preconditionerRank = 100, preconditionerMemoryBytes = 0))(spark)
    val system = factory.build(matrix(b), 2, None)
    try {
      intercept[IllegalStateException](system.solve(new DenseVector(Array(1.0, 3.0))))
      assert(factory.maximumRank == 0)
      intercept[IllegalArgumentException](system.solve(new DenseVector(Array(Double.NaN, 1.0))))
    } finally system.release()
  }

  test("adaptive rank escalation solves and checks the true residual after restarting") {
    val b = BDM((1.0, 2.0), (2.0, -0.5), (0.0, 1.0))
    val factory = new newton.CgFactory(1e-12, 1)(spark)
    val system = factory.build(matrix(b), 2, None)
    val rhs = BDV(1.0, 3.0)
    try {
      val solution = new BDV(system.solve(new DenseVector(rhs.toArray)).values)
      val g = (b.t * b) / (1.0 + 1e-8) + 1e-8 * BDM.eye[Double](2)

      assert(norm(rhs - g * solution) < 1e-12 * norm(rhs))
      assert(factory.maximumRank == 2)
      assert(factory.innerIterations > 1)
    } finally system.release()
  }

  test("matrix-free LP agrees with Cholesky on ill-conditioned and degenerate feasible cases") {
    implicit val session: SparkSession = spark
    val cases = Seq(
      (BDM((1.0, 1.0), (1.0, 1.0001), (0.0, 0.0001)), Array(1.0, 2.0, 3.0), Array(1.0, 1.00005)),
      (BDM((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)), Array(1.0, 1.0, 0.0, 2.0), Array(1.0, 0.0)))
    cases.foreach { case (b, costs, rhs) =>
      val results = Seq(NewtonSolver.Cholesky, NewtonSolver.ConjugateGradient).map { backend =>
        val result = LP.solveSummary(vector(costs), matrix(b), new DenseVector(rhs), solver = backend, maxIter = 100)
        try {
          assert(result.termination == LP.Termination.Converged, s"$backend: $result")
          assert(result.primalResidual < 1e-8)
          assert(result.dualResidual < 1e-8)
          assert(result.dualityGap < 1e-8)
          assert(norm(b.t * local(result.x) - new BDV(rhs)) / (1.0 + norm(new BDV(rhs))) < 1e-8)
          result.objectiveValue
        } finally result.x.unpersist()
      }

      assert(math.abs(results(0) - results(1)) < 1e-6)
    }
  }

  test("dependent feasible rows work with regularization and original residuals still gate convergence") {
    implicit val session: SparkSession = spark
    val rows = matrix(BDM((1.0, 1.0), (1.0, 1.0)))

    val result = LP.solveSummary(vector(Array(1.0, 2.0)), rows, new DenseVector(Array(1.0, 1.0)),
      solver = NewtonSolver.ConjugateGradient)
    try {
      assert(result.termination == LP.Termination.Converged)
      assert(math.abs(result.objectiveValue - 1.0) < 1e-7)
    } finally result.x.unpersist()

    val limited = LP.solveSummary(vector(Array(1.0, 2.0)), rows, new DenseVector(Array(1.0, 1.0)),
      solver = NewtonSolver.ConjugateGradient, maxIter = 1, cgConfig = CgConfig(100.0, 100.0))
    try {
      assert(limited.termination == LP.Termination.IterationLimit)
      assert(limited.primalResidual > 1e-8 || limited.dualResidual > 1e-8 || limited.dualityGap > 1e-8)
    } finally limited.x.unpersist()
  }

  test("matrix-free systems exceed the packed Gramian dimension limit with bounded factor storage") {
    val m = 65536
    val rows = sc.parallelize(Seq(Vectors.sparse(m, Seq(0 -> 1.0))), 1)
    val factory = new newton.CgFactory(1e-12, 20, config = CgConfig(
      preconditionerRank = m, preconditionerMemoryBytes = 32L * m * 2))(spark)
    val system = factory.build(rows, m, None)
    val rhs = new Array[Double](m)
    rhs(0) = 1.0
    rhs(m - 1) = 1e-8
    try {
      val solution = system.solve(new DenseVector(rhs)).values

      assert(math.abs(solution(0) - 1.0 / (1.0 / (1.0 + 1e-8) + 1e-8)) < 1e-10)
      assert(math.abs(solution(m - 1) - 1.0) < 1e-10)
      assert(factory.maximumRank <= 2)
      assert(solution.slice(1, m - 1).forall(_ == 0.0))
    } finally system.release()
  }

  test("matrix-free unbounded reporting is backed by an original-LP certificate") {
    implicit val session: SparkSession = spark

    val result = LP.solveSummary(vector(Array(-1.0, 0.0)), matrix(new BDM(2, 1, Array(1.0, -1.0))),
      new DenseVector(Array(1.0)), solver = NewtonSolver.ConjugateGradient)
    try {
      assert(result.termination == LP.Termination.DualInfeasible)
      val ray = local(result.dualCertificate.get)

      assert(ray.toArray.forall(_ >= 0.0))
      assert(math.abs(ray(0) - 1.0) < 1e-10)
      assert(math.abs(ray(0) - ray(1)) <= 1e-8)
    } finally result.x.unpersist()
  }
}
