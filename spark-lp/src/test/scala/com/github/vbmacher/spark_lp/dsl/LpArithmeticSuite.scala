package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import org.scalatest.funsuite.AnyFunSuite

class LpArithmeticSuite extends AnyFunSuite {
  test("constant dot products retain signs and support division by a negative scalar") {
    val coefficients = Seq(2.0, -3.0)
    val expressions = Seq(LpExpr.constant(4.0), LpExpr.constant(5.0))

    val product = lpDot(coefficients, expressions)
    val divided = product / -2.0

    assert(product.constant == -7.0)
    assert(divided.constant == 3.5)
  }

  test("the dot product of two empty inputs is zero") {
    val coefficients = Seq.empty[Double]
    val expressions = Seq.empty[LpExpr]

    val product = lpDot(coefficients, expressions)

    assert(product.constant == 0.0)
  }

  test("dot products reject either input being longer than the other") {
    val coefficient = Seq(1.0)
    val expression = Seq(LpExpr.zero)

    intercept[LpModelException](lpDot(coefficient, Seq.empty[LpExpr]))
    intercept[LpModelException](lpDot(Seq.empty[Double], expression))
  }

  test("division rejects zero and non-finite divisors") {
    val expression = LpExpr.constant(1.0)
    val invalidDivisors = Seq(0.0, -0.0, Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity)

    invalidDivisors.foreach { divisor =>
      withClue(s"divisor = $divisor: ") {
        intercept[LpModelException](expression / divisor)
      }
    }
  }

  test("dot products reject non-finite coefficients") {
    val expressions = Seq(LpExpr.constant(1.0))
    val invalidCoefficients = Seq(Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity)

    invalidCoefficients.foreach { coefficient =>
      withClue(s"coefficient = $coefficient: ") {
        intercept[LpModelException](lpDot(Seq(coefficient), expressions))
      }
    }
  }
}
