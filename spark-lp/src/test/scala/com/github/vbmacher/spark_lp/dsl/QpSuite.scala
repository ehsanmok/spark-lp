package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.scalatest.funsuite.AnyFunSuite

class QpSuite extends AnyFunSuite with DataFrameSuiteBase {

  test("weighted deviations have independently checked equality KKT witnesses on both backends") {
    implicit val ss: SparkSession = spark
    Seq(NewtonSolver.Cholesky, NewtonSolver.ConjugateGradient).foreach { backend =>
      val model = LpProblem("weighted")
      val x = model.variable("x")
      val y = model.variable("y")
      model += QpObjective.squaredDeviation(x, 2.0) + QpObjective.squaredDeviation(y, 4.0, 2.0)
      model += (x + y === 3.0)

      val result = model.solve(SolveConfig(newtonSolver = backend, maxIterations = 100))
      try {
        assert(result.status == LpStatus.Optimal)
        val xValue = result.value(x)
        val yValue = result.value(y)
        // x=0, y=3, equality multiplier=-4; gradients are both -4, bound multiplier=0.

        assert(math.abs(xValue) < 2e-4)
        assert(math.abs(yValue - 3.0) < 2e-4)
        val boundMultiplier = 2 * (xValue - 2) - 4 * (yValue - 4)

        assert(boundMultiplier >= -1e-7)
        assert(math.abs(xValue * boundMultiplier) < 1e-6)
        assert(math.abs(result.objectiveValue - 6.0) < 1e-6)
        assert(result.residuals.dual < 1e-8)
        assert(result.residuals.gap < 1e-8)
      } finally result.close()
    }
  }

  test("shifts, fixed values, upper bounds and concave maximization preserve objective constants") {
    implicit val ss: SparkSession = spark
    for (backend <- Seq(NewtonSolver.Cholesky, NewtonSolver.ConjugateGradient); sense <- Seq(Minimize, Maximize)) {
      val model = LpProblem("transforms", sense)
      val x = model.variable("x", lowerBound = -3, upperBound = Some(1.0))
      val fixed = model.variable("fixed", lowerBound = 5, upperBound = Some(5.0))
      val objective = QpObjective.squaredDeviation(x, 2) + QpObjective.squaredDeviation(fixed, 3, 2)
      model += objective * (if (sense == Minimize) 1.0 else -1.0)

      val result = model.solve(SolveConfig(newtonSolver = backend, maxIterations = 100))
      try {
        assert(result.status == LpStatus.Optimal)
        assert(math.abs(result.value(x) - 1.0) < 1e-6)
        assert(result.value(fixed) == 5.0)
        assert(math.abs(result.objectiveValue - (if (sense == Minimize) 9.0 else -9.0)) < 1e-6)
      } finally result.close()
    }
  }

  test("unconstrained separable QP and keyed curvature remain distributed") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("keyed")
    val x = model.variables("x", Seq(("a", 2.0, -4.0), ("b", 4.0, -12.0)).toDF("key", "q", "c"), col("key"))
    model += QpObjective.separable(x.sum(col("q")), x.sum(col("c")))

    val result = model.solve()
    try {
      assert(result.status == LpStatus.Optimal)
      val values = result.values(x).select("key", "lp_value").collect().map(r => r.getString(0) -> r.getDouble(1)).toMap

      assert(math.abs(values("a") - 2) < 1e-6)
      assert(math.abs(values("b") - 3) < 1e-6)
      assert(math.abs(result.objectiveValue + 22) < 1e-6)
    } finally result.close()
  }

  test("zero curvature agrees with LP and rejects invalid curvature, categories and curved free variables") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("zero")
    val x = model.variable("x")
    model += QpObjective.separable(0.0 * x, x)
    model += (x >= 2.0)

    val qp = model.solve()
    model.setObjective(x)

    val lp = model.solve()
    try {
      assert(qp.status == lp.status)
      assert(math.abs(qp.objectiveValue - lp.objectiveValue) < 1e-9)
    } finally { qp.close(); lp.close() }
    for ((q, lower, cat) <- Seq((-1.0, 0.0, Continuous), (Double.NaN, 0.0, Continuous),
      (1.0, Double.NegativeInfinity, Continuous), (1.0, 0.0, Integer), (1.0, 0.0, Binary))) {
      val invalid = LpProblem("invalid")
      val v = invalid.variable("v", lowerBound = lower, category = cat)
      invalid += QpObjective.separable(q * v, v)

      intercept[LpModelException](invalid.solve())
    }
  }

  test("positive curvature prevents an LP-only false unbounded certificate") {
    implicit val ss: SparkSession = spark
    Seq(NewtonSolver.Cholesky, NewtonSolver.ConjugateGradient).foreach { backend =>
      val model = LpProblem("bounded quadratic")
      val x = model.variable("x")
      model += QpObjective.separable(2.0 * x, -100.0 * x)
      model += (x >= 1.0)

      val result = model.solve(SolveConfig(newtonSolver = backend, maxIterations = 100))
      try {
        assert(result.status == LpStatus.Optimal)
        assert(math.abs(result.value(x) - 50) < 1e-5)
        assert(math.abs(result.objectiveValue + 2500) < 1e-5)
      } finally result.close()
    }
  }
  test("all-fixed quadratic objective is evaluated without invoking Newton") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("fixed")
    val x = model.variable("x", lowerBound = 5, upperBound = Some(5.0))
    model += QpObjective.squaredDeviation(x, 2, 3)

    val result = model.solve()
    try {
      assert(result.status == LpStatus.Optimal)
      assert(result.iterations == 0)
      assert(result.objectiveValue == 27.0)
    } finally result.close()
  }

  test("curvature-free recession component can still certify unboundedness") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("flat recession")
    val x = model.variable("x")
    val y = model.variable("y")
    model += QpObjective.separable(2.0 * x, -1.0 * y)
    model += (x === 0.0)
    model += (y >= 1.0)

    val result = model.solve(SolveConfig(tolerance = 1e-4, maxIterations = 100))
    try {
      assert(Set[LpStatus](LpStatus.Unbounded, LpStatus.InfeasibleOrUnbounded)(result.status))
    } finally result.close()
  }

}
