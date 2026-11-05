package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.TestingUtils._
import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.funsuite.AnyFunSuite

/**
  * DataFrame-native bulk constraints (`terms` + `lpSumBy` + DataFrame/scalar RHS).
  *
  * Fixture: three shipping arcs into two markets,
  * maximize 3*a1 + 1*a2 + 3*a3 subject to per-market volume capacity:
  * m1: 2*a1 + 1*a2 <= 10, m2: 1*a3 <= 5. Optimum: a1 = 5, a2 = 0, a3 = 5, profit 30.
  */
class LpDslBulkSuite extends AnyFunSuite with DataFrameSuiteBase {

  test("grouped constraint with a DataFrame RHS expands one named row per group") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val capacities = Seq(("m1", 10.0), ("m2", 5.0)).toDF("market", "rhs")

    val arcRows = Seq(
      ("a1", "m1", 2.0, 3.0),
      ("a2", "m1", 1.0, 1.0),
      ("a3", "m2", 1.0, 3.0)).toDF("arc", "market", "vol", "profit")
    val model = LpProblem("transport", Maximize)
    val shipment = model.variables("shipment", arcRows, key = $"arc")
    model += lpSum(shipment * $"profit")
    val shippedByMarket = lpSumBy(
      shipment.terms(source = arcRows, by = Seq($"market"), coefficient = $"vol"),
      by = Seq("market"))
    model += (shippedByMarket <= capacities).named("cap")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 30.0 absTol 1e-4)

    val values = solution.values(shipment).select("arc", "lp_value").collect()
      .map(r => r.getString(0) -> r.getDouble(1)).toMap

    assert(values("a1") ~== 5.0 absTol 1e-4)
    assert(values("a2") ~== 0.0 absTol 1e-4)
    assert(values("a3") ~== 5.0 absTol 1e-4)

    val rows = solution.constraints.collect().map(r => r.getString(0) -> r).toMap

    assert(rows.keySet == Set("cap[m1]", "cap[m2]"))
    val m1 = rows("cap[m1]")

    assert(m1.getString(1) == "m1", "group column carries the group key")
    assert(m1.getString(3) == "<=")
    assert(m1.getDouble(4) == 10.0)
    assert(m1.getDouble(2) ~== 10.0 absTol 1e-4)
    assert(m1.getDouble(5) ~== 0.0 absTol 1e-4)
  }

  test("grouped constraint with a scalar RHS applies the same bound to every group") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val arcRows = Seq(
      ("a1", "m1", 2.0, 3.0),
      ("a2", "m1", 1.0, 1.0),
      ("a3", "m2", 1.0, 3.0)).toDF("arc", "market", "vol", "profit")
    val model = LpProblem("transport", Maximize)
    val shipment = model.variables("shipment", arcRows, key = $"arc")
    model += lpSum(shipment * $"profit")
    val shippedByMarket = lpSumBy(
      shipment.terms(source = arcRows, by = Seq($"market"), coefficient = $"vol"),
      by = Seq("market"))
    model += (shippedByMarket <= 4.0).named("cap")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    // m1: 2*a1 <= 4 -> a1 = 2 (profit 6 beats a2's 4); m2: a3 = 4 (profit 12)

    assert(solution.objectiveValue ~== 18.0 absTol 1e-4)
    val values = solution.values(shipment).select("arc", "lp_value").collect()
      .map(r => r.getString(0) -> r.getDouble(1)).toMap

    assert(values("a1") ~== 2.0 absTol 1e-4)
    assert(values("a3") ~== 4.0 absTol 1e-4)
  }

  test("duplicate RHS rows for one group are rejected") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val capacities = Seq(("m1", 10.0), ("m1", 12.0), ("m2", 5.0)).toDF("market", "rhs")

    val arcRows = Seq(
      ("a1", "m1", 2.0, 3.0),
      ("a2", "m1", 1.0, 1.0),
      ("a3", "m2", 1.0, 3.0)).toDF("arc", "market", "vol", "profit")
    val model = LpProblem("transport", Maximize)
    val shipment = model.variables("shipment", arcRows, key = $"arc")
    model += lpSum(shipment * $"profit")
    val shippedByMarket = lpSumBy(
      shipment.terms(source = arcRows, by = Seq($"market"), coefficient = $"vol"),
      by = Seq("market"))
    model += (shippedByMarket <= capacities).named("cap")

    val error = intercept[LpModelException](model.solve())

    assert(error.getMessage.contains("duplicate RHS"))
    assert(error.getMessage.contains("m1"))
  }

  test("a group with terms but no RHS row is rejected") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val capacities = Seq(("m1", 10.0)).toDF("market", "rhs") // m2 missing

    val arcRows = Seq(
      ("a1", "m1", 2.0, 3.0),
      ("a2", "m1", 1.0, 1.0),
      ("a3", "m2", 1.0, 3.0)).toDF("arc", "market", "vol", "profit")
    val model = LpProblem("transport", Maximize)
    val shipment = model.variables("shipment", arcRows, key = $"arc")
    model += lpSum(shipment * $"profit")
    val shippedByMarket = lpSumBy(
      shipment.terms(source = arcRows, by = Seq($"market"), coefficient = $"vol"),
      by = Seq("market"))
    model += (shippedByMarket <= capacities).named("cap")

    val error = intercept[LpModelException](model.solve())

    assert(error.getMessage.contains("no RHS row"))
    assert(error.getMessage.contains("m2"))
  }

  test("an RHS row with no terms is legal with zero RHS and presolved with a note") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val capacities = Seq(("m1", 10.0), ("m2", 5.0), ("m3", 0.0)).toDF("market", "rhs")

    val arcRows = Seq(
      ("a1", "m1", 2.0, 3.0),
      ("a2", "m1", 1.0, 1.0),
      ("a3", "m2", 1.0, 3.0)).toDF("arc", "market", "vol", "profit")
    val model = LpProblem("transport", Maximize)
    val shipment = model.variables("shipment", arcRows, key = $"arc")
    model += lpSum(shipment * $"profit")
    val shippedByMarket = lpSumBy(
      shipment.terms(source = arcRows, by = Seq($"market"), coefficient = $"vol"),
      by = Seq("market"))
    model += (shippedByMarket === capacities).named("cap")
    // equality capacities: m1 and m2 rows are binding; m3 has no arcs at all

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    val rows = solution.constraints.collect().map(r => r.getString(0) -> r).toMap

    assert(rows.keySet == Set("cap[m1]", "cap[m2]", "cap[m3]"))
    assert(Option(rows("cap[m3]").getString(7)).exists(_.contains("presolved")))
    assert(rows("cap[m1]").getDouble(2) ~== 10.0 absTol 1e-4)
  }

  test("a zero-term row with a non-zero RHS is trivially infeasible and rejected") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val capacities = Seq(("m1", 10.0), ("m2", 5.0), ("m3", 7.0)).toDF("market", "rhs")

    val arcRows = Seq(
      ("a1", "m1", 2.0, 3.0),
      ("a2", "m1", 1.0, 1.0),
      ("a3", "m2", 1.0, 3.0)).toDF("arc", "market", "vol", "profit")
    val model = LpProblem("transport", Maximize)
    val shipment = model.variables("shipment", arcRows, key = $"arc")
    model += lpSum(shipment * $"profit")
    val shippedByMarket = lpSumBy(
      shipment.terms(source = arcRows, by = Seq($"market"), coefficient = $"vol"),
      by = Seq("market"))
    model += (shippedByMarket === capacities).named("cap")

    val error = intercept[LpModelException](model.solve())

    assert(error.getMessage.contains("cap[m3]"))
    assert(error.getMessage.contains("infeasible"))
  }

  test("term rows referencing keys outside the variable domain are rejected") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val model = LpProblem("foreignTerms", Maximize)
    val domain = Seq("a1").toDF("arc")
    val shipment = model.variables("shipment", domain, key = $"arc")
    model += lpSum(shipment)

    val termRows = Seq(("a1", "m1", 1.0), ("zz", "m1", 1.0)).toDF("arc", "market", "vol")
    val grouped = lpSumBy(
      shipment.terms(source = termRows, by = Seq($"market"), coefficient = $"vol"),
      by = Seq("market"))
    model += (grouped <= 5.0).named("cap")

    val error = intercept[LpModelException](model.solve())

    assert(error.getMessage.contains("absent from the domain"))
  }

  test("duplicate (group, variable) term pairs aggregate by summing") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val model = LpProblem("dupTerms", Maximize)
    val domain = Seq("a1").toDF("arc")
    val shipment = model.variables("shipment", domain, key = $"arc")
    model += lpSum(shipment)

    // a1 appears twice in m1's terms: coefficients 1 + 2 = 3
    val termRows = Seq(("a1", "m1", 1.0), ("a1", "m1", 2.0)).toDF("arc", "market", "vol")
    val grouped = lpSumBy(
      shipment.terms(source = termRows, by = Seq($"market"), coefficient = $"vol"),
      by = Seq("market"))
    model += (grouped <= 6.0).named("cap")

    val solution = model.solve()

    assert(solution.status == LpStatus.Optimal)
    assert(solution.objectiveValue ~== 2.0 absTol 1e-4, "3 * a1 <= 6 caps a1 at 2")
  }

  test("terms(...) on a typed variable set without a key column is rejected") {
    implicit val ss: SparkSession = spark
    val ssLocal = spark
    import ssLocal.implicits._
    val model = LpProblem("typedTerms", Minimize)
    val domain = spark.createDataset(Seq(KeyedWeight("a", 1.0)))
    val v = model.variablesOf("v", domain, (k: KeyedWeight) => k.name)
    val termRows = Seq(("a1", "m1", 2.0, 3.0)).toDF("arc", "market", "vol", "profit")

    val error = intercept[LpModelException](v.terms(termRows, Seq($"market"), $"vol"))

    assert(error.getMessage.contains("DataFrame domain"))
  }
}
