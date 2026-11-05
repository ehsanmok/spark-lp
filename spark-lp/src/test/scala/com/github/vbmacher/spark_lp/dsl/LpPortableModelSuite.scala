package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.struct
import org.scalatest.funsuite.AnyFunSuite

class LpPortableModelSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("portable components and a mixed original model reconstruct equivalent algebra and keyed identities") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("portable")
    val xs = model.variables("x", Seq(("a", 1), ("b", 2)).toDF("g", "i"), struct($"g", $"i"), category = Binary)
    val free = model.variable("free", Double.NegativeInfinity)
    val fixed = model.variable("fixed", 2.0, Some(2.0), Integer)
    model += xs.sum + free + fixed + 4.0
    model += (xs.sumBy("g")() >= 1.0).named("minimum")
    model += (free === -1.0).named("free")
    val data = LpPortableModel.fromView(model.inspect)
    val imported = data.toProblem()

    val original = model.solve()
    val copied = imported.model.solve()
    try {
      assert(original.status == LpStatus.Optimal)
      assert(copied.status == LpStatus.Optimal)
      assert(math.abs(copied.objectiveValue - 7.0) < 1e-6)
      val variableData = LpPortableModel.variable(xs(("a", 1)))

      assert(copied.value(imported.variable(variableData.id)) == 1.0)
      assert(imported.constraintIds.size == 3)
      val component = LpPortableModel.expression(xs.sum + 3.0 * free + 8.0)

      assert(imported.expression(component).constant == 8.0)
      assert(imported.expression(component).coefficient(imported.variable(LpPortableModel.variable(free).id)) == 3.0)
      val pending = LpPortableModel.constraint((free <= 0.0).named("pending"))

      assert(imported.constraint(pending).rhs == 0.0)
      val solutionData = LpSolutionData.fromSolution(copied)

      assert(solutionData.values.get.count() == 4)
      assert(solutionData.objective.contains(copied.objectiveValue))
    } finally { original.close(); copied.close() }

    assert(model.constraints.size == 2)
  }

  test("schema, duplicate identities, invalid coefficients and missing references fail explicitly") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("invalid")
    val x = model.variable("x", upperBound = Some(1.0))
    model += lpSum(x)
    model += (x >= 0.0)
    val data = LpPortableModel.fromView(model.inspect)

    intercept[IllegalArgumentException](data.copy(schemaVersion = 2).toProblem())
    intercept[LpModelException](data.copy(variables = data.variables.union(data.variables)).toProblem())
    intercept[LpModelException](data.copy(objective = LpAffineData(spark.sparkContext.parallelize(Seq(
      LpCoefficient(LpVariableId(9, "missing"), 1.0))), 0.0)).toProblem())

    intercept[LpModelException](data.copy(objective = LpAffineData(data.objective.coefficients.map(_.copy(value = Double.NaN)), 0.0)).toProblem())
    intercept[LpModelException](data.toProblem(maxLocalRows = 0))
  }
}
