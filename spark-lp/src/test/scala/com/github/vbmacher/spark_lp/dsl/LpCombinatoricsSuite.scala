package com.github.vbmacher.spark_lp.dsl

import org.scalatest.funsuite.AnyFunSuite

class LpCombinatoricsSuite extends AnyFunSuite {
  import LpCombinatorics._

  test("combinations enumerate fixed-size selections in input position order") {
    val items = Vector("a", "b", "c")

    val selections = combinations(items, 2).toVector

    assert(selections == Vector(Vector("a", "b"), Vector("a", "c"), Vector("b", "c")))
  }

  test("permutations enumerate each selection before moving to the next positions") {
    val items = Vector("a", "b", "c")

    val orderings = permutations(items, 2).toVector

    assert(orderings == Vector(
      Vector("a", "b"), Vector("b", "a"), Vector("a", "c"),
      Vector("c", "a"), Vector("b", "c"), Vector("c", "b")))
  }

  test("equal values at distinct input positions remain distinct selections") {
    val items = Vector("a", "a", "b")

    val selections = combinations(items, 2).toVector

    assert(selections == Vector(Vector("a", "a"), Vector("a", "b"), Vector("a", "b")))
  }

  test("equal values at distinct positions retain both permutations") {
    val items = Vector("a", "a")

    val orderings = permutations(items, 2).toVector

    assert(orderings == Vector(Vector("a", "a"), Vector("a", "a")))
  }

  test("choosing zero items from empty input produces one empty selection") {
    val items = Vector.empty[Int]

    val selections = combinations(items, 0).toVector
    val orderings = permutations(items, 0).toVector

    assert(selections == Vector(Vector.empty[Int]))
    assert(orderings == Vector(Vector.empty[Int]))
  }

  test("a requested size greater than the input yields no fixed-size results") {
    val items = Vector(1)

    val selections = combinations(items, 2).toVector
    val orderings = permutations(items, 2).toVector

    assert(selections.isEmpty)
    assert(orderings.isEmpty)
  }

  test("up-to enumeration caps the size at the number of input items") {
    val items = Vector(1, 2, 3)

    val selections = combinationsUpTo(items, 9).size
    val orderings = permutationsUpTo(items, 9).size

    assert(selections == 8)
    assert(orderings == 16)
  }

  test("negative sizes are rejected by fixed-size and up-to enumeration") {
    val items = Vector(1)
    val invalidSize = -1

    intercept[LpModelException](combinations(items, invalidSize))
    intercept[LpModelException](permutations(items, invalidSize))
    intercept[LpModelException](combinationsUpTo(items, invalidSize))
    intercept[LpModelException](permutationsUpTo(items, invalidSize))
  }

  test("huge result spaces support consuming only a two-result prefix") {
    val items = 0 until 100
    val selectionSize = 50

    val selections = combinations(items, selectionSize).take(2).size
    val orderings = permutations(items, selectionSize).take(2).size

    assert(selections == 2)
    assert(orderings == 2)
  }
}
