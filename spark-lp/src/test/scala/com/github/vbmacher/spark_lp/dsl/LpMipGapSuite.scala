package com.github.vbmacher.spark_lp.dsl

import org.scalatest.funsuite.AnyFunSuite

class LpMipGapSuite extends AnyFunSuite {
  test("either absolute or relative gap tolerance can accept a finite bound") {
    val cases = Seq(
      ("absolute threshold", 10.0, 9.0, 10.0, 0.0, 1.0, true),
      ("negative objective scale", 10.0, 9.0, -100.0, 0.01, 0.0, true),
      ("both thresholds", 10.0, 9.0, 100.0, 0.01, 1.0, true),
      ("neither threshold", 10.0, 9.0, 0.0, 0.1, 0.5, false),
      ("absolute gap too large", 100.0, 90.0, 100.0, 0.0, 1.0, false))

    cases.foreach { case (scenario, incumbent, bound, objective, relative, absolute, expected) =>
      withClue(s"$scenario: ") {
        val config = MipConfig(gapTolerance = relative, absoluteGapTolerance = absolute)

        val accepted = MipGap.accepted(incumbent, bound, objective, config)

        assert(accepted == expected)
      }
    }
  }

  test("an unavailable bound cannot satisfy even a large absolute gap tolerance") {
    val unavailableBounds = Seq(Double.NaN, Double.NegativeInfinity, Double.PositiveInfinity)
    val config = MipConfig(absoluteGapTolerance = 1e9)

    unavailableBounds.foreach { bound =>
      val accepted = MipGap.accepted(1.0, bound, 1.0, config)

      assert(!accepted, s"unavailable bound $bound")
    }
  }

  test("absolute gap tolerance rejects negative and non-finite values") {
    val invalidTolerances = Seq(-1.0, Double.NaN, Double.PositiveInfinity)

    invalidTolerances.foreach { tolerance =>
      withClue(s"tolerance = $tolerance: ") {
        intercept[IllegalArgumentException](MipConfig(absoluteGapTolerance = tolerance))
      }
    }
  }
}
