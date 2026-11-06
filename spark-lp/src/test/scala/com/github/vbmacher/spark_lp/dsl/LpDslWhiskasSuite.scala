package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.TestingUtils._
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

/**
  * Ingredient row used by the Whiskas blending model tests.
  *
  * @param name ingredient key.
  * @param cost cost per gram.
  * @param protein grams of protein per gram of ingredient.
  * @param fat grams of fat per gram of ingredient.
  * @param fibre grams of fibre per gram of ingredient.
  * @param salt grams of salt per gram of ingredient.
  */
case class WhiskasIngredient(
  name: String,
  cost: Double,
  protein: Double,
  fat: Double,
  fibre: Double,
  salt: Double)

/**
  * End-to-end tests on the PuLP Whiskas blending problem:
  * minimize 0.013 chicken + 0.008 beef subject to
  * chicken + beef == 100, protein >= 8, fat >= 6, fibre <= 2, salt <= 0.4.
  * Optimum: chicken = 33.333..., beef = 66.666..., cost = 0.966666...
  */
class LpDslWhiskasSuite extends AnyFunSuite with DataFrameSuiteBase {

  test("untyped DataFrame domain solves Whiskas to optimality") {
    implicit val ss: SparkSession = spark
    val ingredientRows = Seq(
      WhiskasIngredient("chicken", 0.013, 0.100, 0.080, 0.001, 0.002),
      WhiskasIngredient("beef", 0.008, 0.200, 0.100, 0.005, 0.005))
    val localSpark = spark
    import localSpark.implicits._
    val ingredients = ingredientRows.toDF()

    val model = LpProblem("Whiskas", Minimize)
    val amount = model.variables("amount", domain = ingredients, key = $"name")

    model += lpSum(amount * $"cost")
    model += (lpSum(amount) === 100.0).named("total_weight")
    model += (lpSum(amount * $"protein") >= 8.0).named("protein_min")
    model += (lpSum(amount * $"fat") >= 6.0).named("fat_min")
    model += (lpSum(amount * $"fibre") <= 2.0).named("fibre_max")
    model += (lpSum(amount * $"salt") <= 0.4).named("salt_max")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.iterations > 0)
    assert(solution.residuals.primal < 1e-8)
    assert(solution.residuals.dual < 1e-8)
    assert(solution.residuals.gap < 1e-8)
    assert(solution.objectiveValue ~== 0.9666666667 absTol 1e-6)

    val values = solution.values(amount)
      .select("name", "lp_variable", "lp_value")
      .collect()
      .map(r => (r.getString(0), (r.getString(1), r.getDouble(2))))
      .toMap

    assert(values("chicken")._1 == "amount[chicken]")
    assert(values("beef")._1 == "amount[beef]")
    assert(values("chicken")._2 ~== 33.3333333 absTol 1e-4)
    assert(values("beef")._2 ~== 66.6666667 absTol 1e-4)
  }

  test("constraints DataFrame reports original rows and LP shadow prices") {
    implicit val ss: SparkSession = spark
    val ingredientRows = Seq(
      WhiskasIngredient("chicken", 0.013, 0.100, 0.080, 0.001, 0.002),
      WhiskasIngredient("beef", 0.008, 0.200, 0.100, 0.005, 0.005))
    val localSpark = spark
    import localSpark.implicits._
    val ingredients = ingredientRows.toDF()

    val model = LpProblem("Whiskas", Minimize)
    val amount = model.variables("amount", domain = ingredients, key = $"name")

    model += lpSum(amount * $"cost")
    model += (lpSum(amount) === 100.0).named("total_weight")
    model += (lpSum(amount * $"protein") >= 8.0).named("protein_min")
    model += (lpSum(amount * $"fat") >= 6.0).named("fat_min")
    model += (lpSum(amount * $"fibre") <= 2.0).named("fibre_max")
    model += (lpSum(amount * $"salt") <= 0.4).named("salt_max")

    val solution = model.solve()

    assert(solution.constraints.columns.toSeq ==
      Seq("name", "group", "activity", "sense", "rhs", "slack", "dual", "note", "dual_note"))
    val rows = solution.constraints.collect().map(r => r.getString(0) -> r).toMap

    assert(rows.keySet == Set("total_weight", "protein_min", "fat_min", "fibre_max", "salt_max"))

    val total = rows("total_weight")

    assert(total.getString(3) == "==")
    assert(total.getDouble(4) == 100.0)
    assert(total.getDouble(2) ~== 100.0 absTol 1e-6)
    assert(!total.isNullAt(6), "an emitted optimal LP row has a shadow price")

    // salt is the binding <= constraint: activity == rhs, slack ~ 0
    val salt = rows("salt_max")

    assert(salt.getString(3) == "<=")
    assert(salt.getDouble(2) ~== 0.4 absTol 1e-6)
    assert(salt.getDouble(5) ~== 0.0 absTol 1e-6)

    // protein is slack: activity = 0.1*33.33 + 0.2*66.67 = 16.67, slack = activity - rhs for >=
    val protein = rows("protein_min")

    assert(protein.getString(3) == ">=")
    assert(protein.getDouble(2) ~== 16.6666667 absTol 1e-4)
    assert(protein.getDouble(5) ~== 8.6666667 absTol 1e-4)
  }

  test("typed variablesOf + weightedBy solves the identical model") {
    implicit val ss: SparkSession = spark
    import ss.implicits._
    val ingredientRows = Seq(
      WhiskasIngredient("chicken", 0.013, 0.100, 0.080, 0.001, 0.002),
      WhiskasIngredient("beef", 0.008, 0.200, 0.100, 0.005, 0.005))
    val ingredients = spark.createDataset(ingredientRows)

    val model = LpProblem("WhiskasTyped", Minimize)
    val amount = model.variablesOf("amount", ingredients, (i: WhiskasIngredient) => i.name)

    model += lpSum(amount.weightedBy(ingredients)(_.cost))
    model += (lpSum(amount) === 100.0).named("total_weight")
    model += (lpSum(amount.weightedBy(ingredients)(_.protein)) >= 8.0).named("protein_min")
    model += (lpSum(amount.weightedBy(ingredients)(_.fat)) >= 6.0).named("fat_min")
    model += (lpSum(amount.weightedBy(ingredients)(_.fibre)) <= 2.0).named("fibre_max")
    model += (lpSum(amount.weightedBy(ingredients)(_.salt)) <= 0.4).named("salt_max")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 0.9666666667 absTol 1e-6)

    // values() restores the full typed domain columns plus lp_variable/lp_value
    val values = solution.values(amount)

    assert(values.columns.toSeq ==
      Seq("name", "cost", "protein", "fat", "fibre", "salt", "lp_variable", "lp_value"))
    val byName = values.select("name", "lp_value").collect()
      .map(r => r.getString(0) -> r.getDouble(1)).toMap

    assert(byName("chicken") ~== 33.3333333 absTol 1e-4)
    assert(byName("beef") ~== 66.6666667 absTol 1e-4)
  }

  test("scalar variables express the same model, and value() reads one variable") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("WhiskasScalar", Minimize)
    val chicken = model.variable("chicken")
    val beef = model.variable("beef")

    model += 0.013 * chicken + 0.008 * beef
    model += (chicken + beef === 100.0).named("total_weight")
    model += (0.100 * chicken + 0.200 * beef >= 8.0).named("protein_min")
    model += (0.080 * chicken + 0.100 * beef >= 6.0).named("fat_min")
    model += (0.001 * chicken + 0.005 * beef <= 2.0).named("fibre_max")
    model += (0.002 * chicken + 0.005 * beef <= 0.4).named("salt_max")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 0.9666666667 absTol 1e-6)
    assert(solution.value(chicken) ~== 33.3333333 absTol 1e-4)
    assert(solution.value(beef) ~== 66.6666667 absTol 1e-4)
  }

  test("expression algebra: repeated terms sum, constants move across ===, expression RHS normalises") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("algebra", Minimize)
    val x = model.variable("x")
    val y = model.variable("y")

    model += lpSum(Seq(x + 2.0 * y): Iterable[LpExpr])
    // 2x + x === 9  =>  x = 3
    model += (2.0 * x + x === 9.0).named("tripled")
    // y - 3 >= x - 2  =>  y - x >= 1  =>  y = 4 at the minimum
    model += ((y - 3.0) >= (x - 2.0)).named("shifted")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.value(x) ~== 3.0 absTol 1e-5)
    assert(solution.value(y) ~== 4.0 absTol 1e-5)
    assert(solution.objectiveValue ~== 11.0 absTol 1e-5)
  }
}
