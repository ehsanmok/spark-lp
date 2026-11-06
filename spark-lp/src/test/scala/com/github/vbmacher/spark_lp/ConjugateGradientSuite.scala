package com.github.vbmacher.spark_lp

import org.apache.spark.wrappers.{CholeskyDecomposition, ConjugateGradient}
import org.scalatest.funsuite.AnyFunSuite

class ConjugateGradientSuite extends AnyFunSuite {
  private def product(x: Array[Double]): Array[Double] =
    Array(4.0 * x(0) + x(1), x(0) + 3.0 * x(1))

  test("BLAS CG matches LAPACK and preserves the right-hand side and warm start") {
    val rhs = Array(1.0, 2.0)
    val initial = Array(0.1, 0.2)

    val expected = CholeskyDecomposition.solve(Array(4.0, 1.0, 3.0), rhs.clone())
    val cg = new ConjugateGradient(rhs, product, 1e-12, initialGuess = Some(initial))
    cg.solve(_.clone(), 10)

    assert(cg.converged)
    assert(cg.solution.zip(expected).forall { case (a, b) => math.abs(a - b) < 1e-12 })
    val actualResidual = product(cg.solution).zip(rhs).map { case (a, b) => (a - b) * (a - b) }.sum

    assert(math.sqrt(actualResidual) <= cg.targetNorm)
    assert(rhs.sameElements(Array(1.0, 2.0)))
    assert(initial.sameElements(Array(0.1, 0.2)))
    val snapshot = cg.solution
    snapshot(0) = 100.0

    assert(math.abs(cg.solution(0) - expected(0)) < 1e-12)
  }

  test("a stronger preconditioner resumes an exhausted solve with cumulative counters") {
    val cg = new ConjugateGradient(Array(1.0, 2.0), product, 1e-12)
    cg.solve(_.clone(), 1)

    assert(!cg.converged)
    assert(cg.iterations == 1)
    val factor = CholeskyDecomposition.factor(Array(4.0, 1.0, 3.0), 2)
    cg.solve(r => CholeskyDecomposition.solveFactored(factor, 2, r.clone()), 1)

    assert(cg.converged)
    assert(cg.iterations == 2)
    assert(cg.restarts == 1)
  }

  test("zero RHS and an exact warm start require no CG iterations") {
    val zero = new ConjugateGradient(Array(0.0, 0.0), _ => fail("Zero RHS needs no product"),
      1e-12, initialGuess = Some(Array(1.0, 1.0)))
    zero.solve(_.clone(), 10)

    assert(zero.converged)
    assert(zero.iterations == 0)
    assert(zero.solution.sameElements(Array(0.0, 0.0)))
    val warm = new ConjugateGradient(Array(6.0, 7.0), product, 1e-12,
      initialGuess = Some(Array(1.0, 2.0)))
    warm.solve(_ => fail("Exact warm start needs no preconditioner"), 10)

    assert(warm.converged)
    assert(warm.iterations == 0)
  }

  test("CG rejects nonpositive and non-finite curvature") {
    Seq(-1.0, Double.NaN, Double.PositiveInfinity).foreach { curvature =>
      val cg = new ConjugateGradient(Array(1.0), x => Array(curvature * x(0)), 1e-12)

      intercept[IllegalStateException](cg.solve(_.clone(), 10))
    }
  }

  test("absolute tolerance uses the true initial residual") {
    val cg = new ConjugateGradient(Array(1e-8, 0.0), product, 1e-12, absoluteTolerance = 1e-7)
    cg.solve(_ => fail("Initial residual already meets tolerance"), 10)

    assert(cg.converged)
    assert(cg.iterations == 0)
  }

  test("component tolerances keep a large RHS from hiding an unsolved small row") {
    val tolerances = Array(100.0, 1e-8)
    val cg = new ConjugateGradient(Array(1e12, 100.0), _.clone(), 1e-10,
      initialGuess = Some(Array(1e12, 0.0)), componentTolerances = Some(tolerances))
    tolerances(1) = 100.0
    assert(!cg.converged)

    cg.solve(_.clone(), 2)

    assert(cg.converged)
    assert(cg.solution.sameElements(Array(1e12, 100.0)))
    assert(cg.iterations == 1)
  }
}
