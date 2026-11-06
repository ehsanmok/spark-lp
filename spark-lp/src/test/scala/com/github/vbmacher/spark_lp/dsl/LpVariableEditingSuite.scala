package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.SolveControl
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpVariableEditingSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("scalar edits, repeated fixes and unfix restore bounds without changing expression identity") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("editing", Maximize)
    val x = model.variable("x", upperBound = Some(4.0))
    val expression = 2.0 * x + 1.0
    model += expression

    val first = model.solve()
    try {
      // Repeated fixes replace the fixed value while remembering the edited bounds.
      x.setBounds(-1.0, Some(3.0)).fix(2.0).fix(1.0).rename("renamed")

      val fixed = model.solve()
      try {
        assert(fixed.value(x) == 1.0)
        assert(fixed.objectiveValue == 3.0)
      } finally fixed.close()
      // Unfix restores the bounds preceding the first fix.
      x.unfix()

      assert(x.lowerBound == -1.0)
      assert(x.upperBound.contains(3.0))

      val restored = model.solve()
      try {
        assert(restored.value(x) == 3.0)
        assert(restored.objectiveValue == 7.0)
      } finally restored.close()

      assert(first.value(x) == 4.0)
      assert(first.objectiveValue == 9.0)
      assert(expression.coefficient(x) == 2.0)
      // Edits to a copied variable must leave source bounds intact.
      val copied = model.copy()
      copied.variable(x).fix(-1.0)

      assert(x.lowerBound == -1.0)
      assert(x.upperBound.contains(3.0))

      val independent = copied.model.solve()
      try {
        assert(independent.objectiveValue == -1.0)
      } finally independent.close()
    } finally first.close()
  }

  test("keyed metadata edits are lazy, support looser bounds, and preserve previous result names") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val reads = spark.sparkContext.longAccumulator("edit reads")
    val source = Seq("a", "b").toDS().map { k => reads.add(1L); k }.toDF("key")
    val model = LpProblem("members", Maximize)
    val xs = model.variables("x", source, $"key", upperBound = Some(2.0))
    model += xs.sum
    xs("a").setBounds(-1.0, Some(4.0))

    assert(reads.value == 0L)
    val view = model.inspect

    val first = model.solve()
    try {
      assert(math.abs(first.value(xs("a")) - 4.0) < 1e-6)
      assert(math.abs(first.objectiveValue - 6.0) < 1e-6)
      assert(first.constraints.count() == 0)
      xs("a").fix(1.0).rename("special")
      xs.rename("family")

      val fixed = model.solve()
      try {
        assert(math.abs(fixed.objectiveValue - 3.0) < 1e-6)
        assert(fixed.values(xs).select("lp_variable").as[String].collect().toSet == Set("special", "family[b]"))
      } finally fixed.close()

      assert(first.values(xs).select("lp_variable").as[String].collect().toSet == Set("x[a]", "x[b]"))
      assert(first.rounded().values(xs).select("lp_variable").as[String].collect().toSet == Set("x[a]", "x[b]"))
      assert(view.variables.filter(_.name == "x[a]").first().upper.contains(4.0))
      xs("a").unfix()

      assert(xs("a").upperBound.contains(4.0))
      xs.fix(0.5).unfix()

      assert(xs("a").upperBound.contains(4.0))
      assert(xs("b").upperBound.contains(2.0))
      val data = LpPortableModel.fromView(model.inspect)
      val imported = data.toProblem()

      val solution = imported.model.solve()
      try {
        assert(math.abs(solution.objectiveValue - 6.0) < 1e-6)
      } finally solution.close()

      assert(imported.model.inspect.variables.filter(_.name == "special").first().upper.contains(4.0))
    } finally first.close()
  }

  test("integer domains, duplicate names, absent keys and edits during an active solve are rejected") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("invalid")
    val x = model.variable("x", upperBound = Some(3.0))
    val b = model.variable("binary", category = Binary)
    val i = model.variable("integer", upperBound = Some(2.0), category = Integer)

    intercept[LpModelException](b.fix(0.5))
    intercept[LpModelException](b.setBounds(2.0, Some(3.0)))
    intercept[LpModelException](i.setBounds(0.2, Some(0.3)))
    intercept[LpModelException](x.rename("binary"))
    intercept[LpModelException](x.setBounds(Double.NaN, None))
    intercept[LpModelException](x.setBounds(2.0, Some(1.0)))
    val family = model.variables("family", Seq("a").toDF("key"), $"key")
    family("missing").fix(1.0)

    intercept[LpModelException](model.solve())
    val active = LpProblem("active")
    val y = active.variable("y", upperBound = Some(3.0))
    active += lpSum(y)
    active += (y >= 1.0)
    var checked = false

    val solution = active.solve(SolveConfig(control = SolveControl(onProgress = _ => {
      intercept[LpModelException](y.fix(2.0)); checked = true
    })))
    try {
      assert(checked)
      assert(math.abs(solution.value(y) - 1.0) < 1e-6)
    } finally solution.close()
    y.fix(2.0)
  }
}
