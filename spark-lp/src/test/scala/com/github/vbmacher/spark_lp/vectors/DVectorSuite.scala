package com.github.vbmacher.spark_lp.vectors

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import dvector.implicits._
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.scalatest.funsuite.AnyFunSuite

class DVectorSuite extends AnyFunSuite with DataFrameSuiteBase {
  private def distributed(blocks: Seq[Seq[Double]]): DVector =
    sc.parallelize(blocks.map(values => new DenseVector(values.toArray)), blocks.size)

  test("diagonalProduct scales each matrix row by its corresponding vector coordinate") {
    val matrix: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(4.0, 5.0, 6.0)), 2)
    val scales = distributed(Seq(Seq(2.0), Seq(3.0)))

    val scaledRows = scales.diagonalProduct(matrix).collect().toSeq

    assert(scaledRows == Seq(Vectors.dense(2.0, 4.0, 6.0), Vectors.dense(12.0, 15.0, 18.0)))
  }

  test("combine applies both coefficients across unequal-sized vector blocks") {
    val first = distributed(Seq(Seq(2.0, 3.0), Seq(4.0)))
    val second = distributed(Seq(Seq(5.0, 6.0), Seq(7.0)))

    val combined = first.combine(1.1, 4.0, second).collectElements

    assert(Vectors.dense(combined) == Vectors.dense(22.2, 27.3, 32.4))
  }

  test("combine with unit coefficients preserves the cached second input") {
    val first = distributed(Seq(Seq(2.0, 3.0), Seq(4.0)))
    val second = distributed(Seq(Seq(5.0, 6.0), Seq(7.0))).cache()
    try {
      // Materialize the cache so in-place mutation would affect a later read.
      second.count()

      val combined = first.combine(1.0, 1.0, second).collectElements
      val secondAfterCombination = second.collectElements

      assert(Vectors.dense(combined) == Vectors.dense(7.0, 9.0, 11.0))
      assert(Vectors.dense(secondAfterCombination) == Vectors.dense(5.0, 6.0, 7.0))
    } finally second.unpersist(blocking = false)
  }

  test("dot sums coordinate products across all vector blocks") {
    val first = distributed(Seq(Seq(2.0, 3.0), Seq(4.0)))
    val second = distributed(Seq(Seq(5.0, 6.0), Seq(7.0)))

    val product = first.dot(second)

    assert(product == 56.0)
  }

  test("entrywiseProd multiplies corresponding coordinates across blocks") {
    val first = distributed(Seq(Seq(2.0, 3.0), Seq(4.0)))
    val second = distributed(Seq(Seq(5.0, 6.0), Seq(7.0)))

    val product = first.entrywiseProd(second).collectElements

    assert(Vectors.dense(product) == Vectors.dense(10.0, 18.0, 28.0))
  }

  test("entrywiseNegDiv returns infinity for positive and zero divisors") {
    val numerator = distributed(Seq(Seq(2.0, 3.0), Seq(4.0)))
    val divisor = distributed(Seq(Seq(5.0, -6.0), Seq(0.0)))

    val quotient = numerator.entrywiseNegDiv(divisor).collectElements

    assert(Vectors.dense(quotient) == Vectors.dense(Double.PositiveInfinity, 0.5, Double.PositiveInfinity))
  }

  test("sum includes coordinates from every block") {
    val vector = distributed(Seq(Seq(2.0, 3.0), Seq(4.0)))

    val total = vector.sum()

    assert(total == 9.0)
  }

  test("maxValue searches every block") {
    val vector = distributed(Seq(Seq(2.0, 3.0), Seq(4.0)))

    val maximum = vector.maxValue

    assert(maximum == 4.0)
  }

  test("minValue searches every block") {
    val vector = distributed(Seq(Seq(2.0, 3.0), Seq(4.0)))

    val minimum = vector.minValue

    assert(minimum == 2.0)
  }

  test("extrema skip empty blocks and use infinite sentinels when all blocks are empty") {
    val mixed = distributed(Seq(Seq.empty[Double], Seq(2.0, -3.0)))
    val empty = distributed(Seq(Seq.empty[Double], Seq.empty[Double]))

    val mixedMinimum = mixed.minValue
    val mixedMaximum = mixed.maxValue
    val emptyMinimum = empty.minValue
    val emptyMaximum = empty.maxValue

    assert(mixedMinimum == -3.0)
    assert(mixedMaximum == 2.0)
    assert(emptyMinimum == Double.PositiveInfinity)
    assert(emptyMaximum == Double.NegativeInfinity)
  }
}
