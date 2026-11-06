package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import java.io.StringWriter

class LpAlgebraSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("algebra shows original signs, constants, names, categories and unmistakable truncation") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("render", Maximize)
    val x = model.variable("x", Double.NegativeInfinity)
    val b = model.variable("b", category = Binary)
    model += 2.0 * x - x - 3.0 * b + 0.1234567890123456
    val constraint = (x + b <= 5.0).named("capacity")
    model += constraint
    val writer = new StringWriter()

    LpAlgebra.write(model.inspect, writer)
    val full = writer.toString
    val truncatedPreview = LpAlgebra.preview(model.inspect, 2)
    val completePreview = LpAlgebra.preview(model.inspect, 100)
    val emptyExpression = LpAlgebra.expression(lpSum(Seq.empty[LpExpr]))
    val renderedConstraint = LpAlgebra.constraint(constraint)

    assert(full.contains("model `render`: Maximize"))
    assert(full.contains("+ 1.0 * `x`"))
    assert(full.contains("- 3.0 * `b`"))
    assert(full.contains("0.1234567890123456"))
    assert(full.contains("-inf <= `x` <= +inf [Continuous]"))
    assert(full.contains("[Binary]"))
    assert(full.contains("constraint `capacity`:"))
    assert(truncatedPreview.contains("TRUNCATED:"))
    assert(completePreview.trim == full.trim)
    assert(emptyExpression == "+ 0.0")
    assert(renderedConstraint.contains("<= 5.0"))
    assert(model.constraints.size == 1)
    intercept[IllegalArgumentException](LpAlgebra.preview(model.inspect, -1))
  }

  test("complete rendering includes quadratic factors and grouped keyed identities") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("factors")
    val x = model.variables("x", Seq("a", "b").toDF("key"), $"key")
    model += QpObjective.squared(x.sum - 1.0, 2.0)
    model += (x.sumBy("key")() <= 1.0).named("capacity")
    val writer = new StringWriter()

    LpAlgebra.write(model.inspect, writer)
    val rendered = writer.toString

    assert(rendered.contains("2.0 * ("))
    assert(rendered.contains(")^2"))
    assert(rendered.contains("capacity[a]"))
    assert(rendered.contains("x[b]"))
  }
}
