package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.struct
import org.scalatest.funsuite.AnyFunSuite

class LpModelViewSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("read-only original-model inspection distinguishes declarations, expansion and cancellations") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("inspect", Maximize)
    val reads = spark.sparkContext.longAccumulator("inspection")
    val source = Seq(("a", 1, 2.0), ("b", 2, 3.0)).toDS().map { v => reads.add(1L); v }.toDF("g", "i", "c")
    val family = model.variables("x", source, struct($"g", $"i"), category = Binary)
    val fixed = model.variable("fixed", 5.0, Some(5.0))
    model.variables("empty", Seq.empty[String].toDF("key"), $"key")
    model += fixed + family.sum + family.sum - family.sum + 7.0
    model += (family.sumBy("g")($"c") <= Seq(("a", 2.0), ("b", 3.0), ("c", 0.0)).toDF("g", "rhs")).named("capacity")
    model += (fixed === 5.0).named("fixed")
    val view = model.inspect

    assert(reads.value == 0L)
    assert(view.variableDeclarations.size == 3)
    assert(view.constraintDeclarations.size == 2)
    assert(view.objectiveConstant == 7.0)
    assert(view.sense == Maximize)
    assert(view.statistics() == LpModelStatistics(3, 2, 3L, 4L, 3L, 3L))
    val variables = view.variables.collect()

    assert(variables.count(_.category == Binary) == 2)
    assert(variables.find(_.name == "fixed").exists(v => v.lower == 5.0 && v.upper.contains(5.0)))
    assert(view.coefficients.collect().map(_.value).sorted.toSeq == Seq(1.0, 2.0, 3.0))
    assert(view.constraints.filter(_.name == "capacity[c]").count() == 1)
    // The captured view must remain a snapshot after the source model changes.
    model += (fixed >= 0.0).named("later")
    model.setObjective(2.0 * fixed)

    assert(view.constraintDeclarations.size == 2)
    assert(view.objectiveConstant == 7.0)
    assert(model.inspect.constraintDeclarations.size == 3)
    assert(spark.range(3).count() == 3)
  }

  test("inspection can expose contradictory original rows without invoking presolve or optimization") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("contradiction")
    val x = model.variable("x", lowerBound = 2.0, upperBound = Some(2.0))
    model += (x === 1.0)
    val view = model.inspect

    assert(view.statistics() == LpModelStatistics(1, 1, 1, 1, 1, 0))
    assert(view.constraints.first().rhs == 1.0)
    assert(view.variables.first().lower == 2.0)
  }
}
