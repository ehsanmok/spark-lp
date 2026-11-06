package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpRoundingSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("integer snapping applies only near an integer and only to integral categories") {
    val rule = LpRounding(integerTolerance = 1e-5, boundTolerance = 1e-6)

    val nearInteger = rule(1.9999999, Integer, 0.2, Some(3.8))
    val farFromInteger = rule(1.99, Integer, 0.0, None)
    val continuous = rule(1.9999999, Continuous, 0.0, None)
    val binary = rule(0.9999999, Binary, 0.2, None)

    assert(nearInteger == 2.0)
    assert(farFromInteger == 1.99)
    assert(continuous == 1.9999999)
    assert(binary == 1.0)
  }

  test("continuous values within bound tolerance snap to the declared endpoint") {
    val rule = LpRounding(integerTolerance = 1e-5, boundTolerance = 1e-6)

    val atLowerBound = rule(-2.0 - 1e-7, Continuous, -2.0, None)
    val atUpperBound = rule(3.0 + 1e-7, Continuous, 0.0, Some(3.0))

    assert(atLowerBound == -2.0)
    assert(atUpperBound == 3.0)
  }

  test("rounding rejects negative and non-finite tolerances") {
    val invalidTolerances = Seq(-1.0, Double.NaN, Double.PositiveInfinity)

    invalidTolerances.foreach { tolerance =>
      withClue(s"tolerance = $tolerance: ") {
        intercept[IllegalArgumentException](LpRounding(tolerance))
        intercept[IllegalArgumentException](LpRounding(boundTolerance = tolerance))
      }
    }
  }

  test("integer tolerance cannot make half-integers eligible for snapping") {
    val ambiguousTolerance = 0.5

    intercept[IllegalArgumentException](LpRounding(ambiguousTolerance))
  }

  test("rounded reports may violate constraints while leaving the raw solution unchanged") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("report")
    val x = model.variable("x")
    val family = model.variables("family", Seq("a", "b").toDF("key"), $"key")
    model += x + family.sum
    model += (x >= 1e-4)
    model += (family.sum >= 2e-4)

    val raw = model.solve()
    try {
      val originalObjective = raw.objectiveValue
      val originalCandidate = raw.candidate

      // Presentation snapping deliberately exceeds the model's positive lower limits.
      val report = raw.rounded(LpRounding(boundTolerance = 1e-3))
      val roundedScalar = report.value(x)
      val nonzeroRoundedMembers = report.values(family).filter($"lp_value" =!= 0.0).count()
      val positiveRawMembers = raw.values(family).filter($"lp_value" > 0.0).count()

      assert(roundedScalar == 0.0)
      assert(nonzeroRoundedMembers == 0)
      assert(positiveRawMembers == 2)
      assert(raw.objectiveValue == originalObjective)
      assert(raw.candidate == originalCandidate)
      assert(raw.value(x) > 0.0)

      // The report borrows the raw result's lifetime.
      raw.close()

      intercept[LpModelException](report.value(x))
      intercept[LpModelException](report.values(family))
    } finally raw.close()
  }
}
