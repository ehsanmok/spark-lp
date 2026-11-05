package com.github.vbmacher.spark_lp.vectors

import com.github.vbmacher.spark_lp.TestingUtils._
import com.github.vbmacher.spark_lp.vectors.dense_vector.implicits.DenseVectorOps
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.scalatest.funsuite.AnyFunSuite

class DenseVectorSuite extends AnyFunSuite {
  test("combine subtracts the first vector from the second at the first solver iteration") {
    // Recorded solver inputs retain the numerical regression from the original test.
    val first = new DenseVector(Array(
      1037.8830194476832, 919.2678172250901, 1215.8058227815707, 59.30760111129604, 59.307601111296094))
    val second = new DenseVector(Array(
      130.74622637382816, 109.16609581526643, 124.20404493660249, 10.918025554611186, 11.577626670204133))
    val expected = Vectors.dense(
      -907.1367930738562, -810.1017214098238, -1091.6017778449684, -48.38957555668489, -47.729974441092)

    val combined = first.combine(-1.0, 1.0, second)

    assert(combined ~= expected relTol 1e-6)
  }

  test("dot sums corresponding coordinate products") {
    val first = new DenseVector(Array(2.0, 3.0))
    val second = new DenseVector(Array(5.0, 6.0))

    val product = first.dot(second)

    assert(product == 28.0)
  }

  test("entrywiseProd multiplies corresponding coordinates") {
    val first = new DenseVector(Array(2.0, 3.0))
    val second = new DenseVector(Array(5.0, 6.0))

    val product = first.entrywiseProd(second)

    assert(product == Vectors.dense(10.0, 18.0))
  }

  test("entrywiseNegDiv returns infinity for a nonnegative divisor") {
    val numerator = new DenseVector(Array(2.0, 0.0, 3.0))
    val divisor = new DenseVector(Array(-1.0, 1.0, -3.0))

    val quotient = numerator.entrywiseNegDiv(divisor)

    assert(quotient == Vectors.dense(2.0, Double.PositiveInfinity, 1.0))
  }

  test("sum includes every coordinate and ignores zero contributions") {
    val vector = new DenseVector(Array(1.0, 2.0, 0.0, 3.0))

    val total = vector.sum

    assert(total == 6.0)
  }

  test("maxValue returns the largest coordinate") {
    val vector = new DenseVector(Array(1.0, 2.0, 0.0, 3.0))

    val maximum = vector.maxValue

    assert(maximum == 3.0)
  }

  test("minValue returns the smallest coordinate") {
    val vector = new DenseVector(Array(1.0, 2.0, 3.0))

    val minimum = vector.minValue

    assert(minimum == 1.0)
  }
}
