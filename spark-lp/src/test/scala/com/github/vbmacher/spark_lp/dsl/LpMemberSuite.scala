package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, struct}
import org.scalatest.funsuite.AnyFunSuite

class LpMemberSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("squared deviation targets only the selected member") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("selected curvature")
    val xs = model.variables("x", Seq("a", "b").toDF("key"), $"key", upperBound = Some(3.0))
    model += QpObjective.squaredDeviation(xs("a"), 2.0) + lpSum(xs("b"))

    val result = model.solve()
    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.value(xs("a")) - 2.0) < 1e-5)
      assert(math.abs(result.value(xs("b"))) < 1e-5)
      assert(math.abs(result.objectiveValue) < 1e-5)
    } finally result.close()
  }

  test("member handles mix with scalars and family sums, preserve copies and perform no lookup job") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val reads = spark.sparkContext.longAccumulator("lookup reads")
    val source = Seq(("a", 1), ("b", 2)).toDS().map { v => reads.add(1L); v }.toDF("group", "index")
    val model = LpProblem("members")
    val family = model.variables("x", source, struct(col("group"), col("index")))
    val a = family(("a", 1))

    assert(reads.value == 0L)
    assert(a.selectedKey == family(("a", 1)).selectedKey)
    val scalar = model.variable("s", lowerBound = 1.0, upperBound = Some(1.0))
    model += family.sum + scalar
    model += (a / 2.0 + family(("a", 1)) >= 3.0)
    model += (family(("b", 2)) >= 1.0)
    val copied = model.copy()
    for ((m, av) <- Seq((model, a), (copied.model, copied.variable(a)))) {
      val result = m.solve()
      try {
        assert(result.status == LpStatus.Optimal)
        assert(math.abs(result.objectiveValue - 4.0) < 1e-6)
        assert(math.abs(result.value(av) - 2.0) < 1e-6)
      } finally result.close()
    }
  }

  test("typed primitive and custom keys work; missing, incompatible, null and foreign keys fail") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("typed")
    val family = model.variablesOf("x", Seq(1, 2).toDS(), (x: Int) => x)
    model += family.sum
    model += (family(1) >= 2.0)
    implicit val encoded: LpKeyEncoder[MemberKey] = LpKeyEncoder.instance(k => Seq(k.id))
    model += (family(MemberKey(2)) >= 3.0)

    val result = model.solve()
    try {
      assert(math.abs(result.value(family(MemberKey(2))) - 3.0) < 1e-6)
    } finally result.close()

    intercept[LpModelException](family(null: String))
    for (member <- Seq(family(99), family("1"))) {
      val copy = model.copy()
      copy.model += (copy.variable(member) >= 0.0)

      assert(intercept[LpModelException](copy.model.solve()).getMessage.contains("key"))
    }
    val foreign = LpProblem("foreign")
    foreign += lpSum(family(1))
    foreign += (family(1) >= 0.0)

    intercept[LpModelException](foreign.solve())
  }
}
/**
  * Typed variable-member key used by member-editing tests.
  *
  * @param id encoded member identifier.
  */
final case class MemberKey(id: Int)
