package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class LpLocalVariablesSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("scalar, dictionary and matrix declarations solve the same binary model") {
    implicit val ss: SparkSession = spark
    for (factory <- 0 to 2) {
      val model = LpProblem("local", Maximize)
      val xs = factory match {
        case 0 => Vector(model.variable("x", category = Binary), model.variable("y", category = Binary))
        case 1 =>
          val indexed = model.indexedVariables("v", Vector("a", "b"), category = Binary)

          assert(indexed("a") eq indexed.values.head)
          indexed.values
        case _ => model.matrixVariables("v", Vector(1), Vector("a", "b"), category = Binary).values
      }
      model += (2.0 * xs(0) + xs(1))
      model += (xs(0) + xs(1) <= 1.0)

      val result = model.solve()
      try {
        assert(result.status == LpStatus.Optimal)
        assert(result.objectiveValue == 2.0)
        assert(result.value(xs(0)) == 1.0)
        assert(result.value(xs(1)) == 0.0)
      } finally result.close()
    }
  }

  test("ordering, composite identity, deterministic naming and validation before allocation") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("keys")
    val matrix = model.matrixVariables("m", Seq(2, 1), Seq("b", "a"), -2.0, Some(3.0), Integer)

    assert(matrix.keys == Vector((2, "b"), (2, "a"), (1, "b"), (1, "a")))
    assert(matrix.values.forall(v => v.handle.category == Integer && v.handle.lowerBound == -2.0))
    val another = LpProblem("same").matrixVariables("m", Seq(2, 1), Seq("b", "a"))

    assert(matrix.values.map(_.name) == another.values.map(_.name))
    assert(model.indexedVariables("empty", Seq.empty[String]).size == 0)
    assert(model.matrixVariables("empty", Seq.empty[Int], Seq("x")).size == 0)
    val size = model.handles.size

    intercept[IllegalArgumentException](model.indexedVariables("dup", Seq("a", "a")))
    intercept[IllegalArgumentException](model.indexedVariables("null", Seq("a", null)))
    intercept[IllegalArgumentException](model.matrixVariables("dup", Seq.empty[Int], Seq("a", "a")))
    intercept[IllegalArgumentException](model.matrixVariables("m", Seq(2), Seq("b")))
    assert(model.handles.size == size)
    val composite = model.indexedVariables("parts", Seq(("a,b", "c"), ("a", "b,c")))

    assert(composite.values.map(_.name).distinct.size == 2)
  }
}
