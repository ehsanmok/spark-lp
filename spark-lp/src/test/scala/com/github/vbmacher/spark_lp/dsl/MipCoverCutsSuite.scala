package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.compiler.IntColumn
import org.scalatest.funsuite.AnyFunSuite

class MipCoverCutsSuite extends AnyFunSuite {
  private def columns(weights: Vector[Double]): Vector[IntColumn] = weights.zipWithIndex.map { case (a, i) =>
    IntColumn(i.toLong, i, "", 0.0, 1.0, 0.0, i + 1, Map(0 -> a))
  }
  private val enabled = MipCutsConfig(enabled = true)

  test("global covers preserve every independently enumerated feasible binary assignment") {
    val weights = Vector(2.0, 3.0, 4.0, 5.0, 7.0, 9.0)
    val cols = columns(weights)
    val cuts = MipCoverCuts.generate(cols, Array(12.0), Array.fill(6)(0.0), Array.fill(6)(1.0), Set.empty,
      Map(0L -> 1.0, 1L -> 1.0, 2L -> 1.0, 3L -> 0.6), global = true, enabled, Set.empty, 10)

    assert(cuts.nonEmpty)
    assert(cuts.forall(_.scope == GlobalCut))
    for (mask <- 0 until 64) {
      val bits = weights.indices.map(i => if ((mask & (1 << i)) != 0) 1.0 else 0.0)
      if (weights.zip(bits).map { case (a, b) => a * b }.sum <= 12.0)
        assert(cuts.forall(cut => cut.columns.map(g => bits(g.toInt)).sum <= cut.rhs))
    }
    val repeated = MipCoverCuts.generate(cols, Array(12.0), Array.fill(6)(0.0), Array.fill(6)(1.0), Set.empty,
      Map(0L -> 1.0, 1L -> 1.0, 2L -> 1.0, 3L -> 0.6), global = true, enabled, cuts.map(_.signature).toSet, 10)

    assert(repeated.isEmpty)
  }

  test("local covers apply only within their branch domain") {
    val cols = columns(Vector(2.0, 2.0, 2.0))
    val lower = Array(0.0, 1.0, 0.0)
    val upper = Array(1.0, 1.0, 1.0)
    val cut = MipCoverCuts.generate(cols, Array(1.0), lower, upper, Set.empty,
      Map(0L -> 0.5), global = false, enabled, Set.empty, 10).head

    assert(cut.columns == Vector(0L))
    assert(cut.rhs == 0.0)
    assert(cut.scope.permits(lower, upper))
    assert(cut.scope.permits(Array(0.0, 1.0, 0.0), Array(0.0, 1.0, 1.0)))
    assert(!cut.scope.permits(Array(0.0, 0.0, 0.0), Array(1.0, 0.0, 1.0)))
    intercept[IllegalArgumentException](MipCoverCuts.generate(cols, Array(1.0), lower, upper, Set.empty,
      Map(0L -> 0.5), global = true, enabled, Set.empty, 10))
  }

  test("numerically marginal, unbounded-negative and exhausted-budget deductions are rejected") {
    val cols = columns(Vector(1.0, 1.0))
    val lo = Array(0.0, 0.0)
    val hi = Array(1.0, 1.0)
    val values = Map(0L -> 1.0, 1L -> 1.0)

    assert(MipCoverCuts.generate(cols, Array(2.0 - 1e-12), lo, hi, Set.empty, values, true, enabled, Set.empty, 10).isEmpty)
    assert(MipCoverCuts.generate(cols, Array(1.5), lo, hi, Set(0), values, true, enabled, Set.empty, 10).isEmpty)
    assert(MipCoverCuts.generate(cols, Array(1.5), lo, hi, Set.empty, values, true, enabled, Set.empty, 0).isEmpty)
    assert(MipCoverCuts.generate(columns(Vector(1e-15, 1.0)), Array(0.5), lo, hi, Set.empty, values, true, enabled, Set.empty, 10).isEmpty)
  }

  test("covers remain valid with shifted binaries and negative bounded integer contributions") {
    val random = new scala.util.Random(66L)
    var accepted = 0
    for (_ <- 0 until 100) {
      val weights = Vector.fill(3)(1.0 + random.nextInt(7)) :+ -(1.0 + random.nextInt(3))
      val roots = Vector(-2.0, 1.0, 0.0, -1.0)
      val widths = Vector(1.0, 1.0, 1.0, 3.0)
      val cols = weights.indices.map(i => IntColumn(i.toLong, i, "", roots(i), roots(i) + widths(i),
        0.0, i + 1, Map(0 -> weights(i)))).toVector
      val capacity = random.nextInt(12) - 3.0
      val cuts = MipCoverCuts.generate(cols, Array(capacity), roots.toArray,
        roots.zip(widths).map { case (a, b) => a + b }.toArray, Set.empty,
        Map(0L -> 1.0, 1L -> 1.0, 2L -> 1.0, 3L -> 1.5), true, enabled, Set.empty, 10)
      accepted += cuts.size
      for (a <- 0 to 1; b <- 0 to 1; c <- 0 to 1; d <- 0 to 3) {
        val shifted = Vector(a.toDouble, b.toDouble, c.toDouble, d.toDouble)
        if (weights.zip(shifted).map { case (w, v) => w * v }.sum <= capacity)
          assert(cuts.forall(cut => cut.columns.map(g => shifted(g.toInt)).sum <= cut.rhs))
      }
    }

    assert(accepted > 0)
  }

}
