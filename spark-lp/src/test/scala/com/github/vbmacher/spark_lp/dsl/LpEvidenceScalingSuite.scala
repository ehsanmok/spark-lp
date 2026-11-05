package com.github.vbmacher.spark_lp.dsl

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.funsuite.AnyFunSuite

class LpEvidenceScalingSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("large RHS values cannot turn a feasible row into an infeasibility witness") {
    val key = (0, "x")
    for (rhs <- Seq(1e9, 1e12)) {
      val model = EvidenceModel(Vector(EvidenceRow("pinned", None, "=", rhs)),
        sc.parallelize(Seq(key -> EvidenceVariable("x", 0.0, None, 0.0, Map(0 -> 1.0)))), Minimize)
      val proof = InfeasibilityCertificate(model, Vector(1.0 / rhs),
        sc.parallelize(Seq(key -> (0.0, 0.0))))

      val verified = proof.verify()

      assert(math.abs(verified.margin - 1.0) < 1e-12)
      assert(verified.residual <= 1e-8)
      assert(!verified.valid)
    }
  }

  test("large objective coefficients cannot turn a bounded row into an unbounded ray") {
    val key = (0, "x")
    for (cost <- Seq(1e9, 1e12)) {
      val model = EvidenceModel(Vector(EvidenceRow("pinned", None, "=", 1.0)),
        sc.parallelize(Seq(key -> EvidenceVariable("x", 0.0, None, -cost, Map(0 -> 1.0)))), Minimize)
      val proof = UnboundedDirection(model, sc.parallelize(Seq(key -> (1.0 / cost))),
        Some(sc.parallelize(Seq(key -> 1.0))))

      val verified = proof.verify()

      assert(math.abs(verified.margin - 1.0) < 1e-12)
      assert(verified.residual <= 1e-8)
      assert(!verified.valid)
    }
  }

  test("genuine infeasibility witnesses survive uniform changes in row units") {
    val key = (0, "x")
    for (scale <- Seq(1e-12, 1.0, 1e12)) {
      val model = EvidenceModel(Vector(EvidenceRow("cap", None, "<=", 0.0),
        EvidenceRow("floor", None, ">=", scale)),
        sc.parallelize(Seq(key -> EvidenceVariable("x", Double.NegativeInfinity, None, 0.0,
          Map(0 -> scale, 1 -> scale)))), Minimize)
      val proof = InfeasibilityCertificate(model, Vector(-1.0 / scale, 1.0 / scale),
        sc.parallelize(Seq(key -> (0.0, 0.0))))

      val verified = proof.verify()

      assert(verified.valid)
      assert(math.abs(verified.margin - 1.0) < 1e-12)
    }
  }

  test("genuine rays preserve a pinned variable while objective units change") {
    val pinned = (0, "pinned")
    val free = (1, "free")
    for (scale <- Seq(1e-12, 1.0, 1e12)) {
      val model = EvidenceModel(Vector(EvidenceRow("pin", None, "=", 1e12)),
        sc.parallelize(Seq(
          pinned -> EvidenceVariable("pinned", 0.0, None, 0.0, Map(0 -> 1.0)),
          free -> EvidenceVariable("free", 0.0, None, -scale, Map.empty))), Minimize)
      val proof = UnboundedDirection(model, sc.parallelize(Seq(pinned -> 0.0, free -> (1.0 / scale))),
        Some(sc.parallelize(Seq(pinned -> 1e12, free -> 0.0))))

      val verified = proof.verify()

      assert(verified.valid)
      assert(math.abs(verified.margin - 1.0) < 1e-12)
    }
  }

  test("small positive curvature cannot disappear through objective normalization") {
    val key = (0, "x")
    val model = EvidenceModel(Vector.empty,
      sc.parallelize(Seq(key -> EvidenceVariable("x", 0.0, None, -1e12, Map.empty, curvature = 1e-12))),
      Minimize)
    val proof = UnboundedDirection(model, sc.parallelize(Seq(key -> 1e-12)),
      Some(sc.parallelize(Seq(key -> 0.0))))

    val verified = proof.verify()

    assert(verified.residual <= 1e-8)
    assert(!verified.valid)
  }

  test("unrelated large RHS values do not invalidate exact sparse infeasibility witnesses") {
    val impossible = (0, "impossible")
    val pinned = (1, "pinned")
    val model = EvidenceModel(Vector(EvidenceRow("impossible", None, "=", -1.0),
      EvidenceRow("pin", None, "=", 1e12)), sc.parallelize(Seq(
      impossible -> EvidenceVariable("impossible", 0.0, None, 0.0, Map(0 -> 1.0)),
      pinned -> EvidenceVariable("pinned", 0.0, None, 0.0, Map(1 -> 1.0)))), Minimize)
    val proof = InfeasibilityCertificate(model, Vector(-1.0, 0.0),
      sc.parallelize(Seq(impossible -> (1.0, 0.0), pinned -> (0.0, 0.0))))

    val verified = proof.verify()

    assert(verified.valid)
  }

  test("small essential multipliers survive mixed row units") {
    val key = (0, "x")
    val model = EvidenceModel(Vector(EvidenceRow("first", None, "=", 1.0),
      EvidenceRow("second", None, "=", 2e12)),
      sc.parallelize(Seq(key -> EvidenceVariable("x", Double.NegativeInfinity, None, 0.0,
        Map(0 -> 1.0, 1 -> 1e12)))), Minimize)
    val proof = InfeasibilityCertificate(model, Vector(-1.0, 1e-12),
      sc.parallelize(Seq(key -> (0.0, 0.0))))

    val verified = proof.verify()

    assert(verified.valid)
  }

  test("unrelated large costs do not invalidate exact sparse unbounded rays") {
    val free = (0, "free")
    val pinned = (1, "pinned")
    val model = EvidenceModel(Vector(EvidenceRow("pin", None, "=", 0.0)),
      sc.parallelize(Seq(
        free -> EvidenceVariable("free", 0.0, None, -1.0, Map.empty),
        pinned -> EvidenceVariable("pinned", 0.0, None, 1e12, Map(0 -> 1.0)))), Minimize)
    val proof = UnboundedDirection(model, sc.parallelize(Seq(free -> 1.0, pinned -> 0.0)),
      Some(sc.parallelize(Seq(free -> 0.0, pinned -> 0.0))))

    val verified = proof.verify()

    assert(verified.valid)
  }

  test("tiny bound multipliers cannot hide a false certificate in mixed row units") {
    val first = (0, "first")
    val second = (1, "second")
    val model = EvidenceModel(Vector(EvidenceRow("first", None, "=", 100.0),
      EvidenceRow("second", None, "=", 1e12)), sc.parallelize(Seq(
      first -> EvidenceVariable("first", 0.0, None, 0.0, Map(0 -> 1.0)),
      second -> EvidenceVariable("second", 0.0, None, 0.0, Map(1 -> 1.0)))), Minimize)
    val margin = 9e11
    val proof = InfeasibilityCertificate(model, Vector(-1e9 / margin, 1.0 / margin),
      sc.parallelize(Seq(first -> (1e9 / margin, 0.0), second -> (1e-300, 0.0))))

    val verified = proof.verify()

    assert(verified.residual <= 1e-8)
    assert(!verified.valid)
  }
}
