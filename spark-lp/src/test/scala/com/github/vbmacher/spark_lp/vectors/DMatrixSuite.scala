package com.github.vbmacher.spark_lp.vectors

import com.github.vbmacher.spark_lp.TestingUtils._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import dmatrix.implicits._
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.scalatest.funsuite.AnyFunSuite

class DMatrixSuite extends AnyFunSuite with DataFrameSuiteBase {
  test("Gramian products read each mixed dense and sparse row once despite empty partitions") {
    val reads = sc.longAccumulator("gramian-product-rows")
    val rows: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.sparse(3, Seq(0 -> 4.0, 2 -> 6.0))), 8).map { row =>
      reads.add(1L)
      row
    }
    val weights: DVector = sc.parallelize(Seq(2.0, 3.0), 8).glom().map(new DenseVector(_))
    val direction = sc.broadcast(new DenseVector(Array(1.0, -2.0, 3.0)))
    try {
      val product = rows.gramianProduct(direction)

      assert(product ~== Vectors.dense(94.0, 12.0, 150.0) absTol 1e-12)
      assert(reads.value == 2L)

      // Reset only the instrumentation before exercising the weighted path.
      reads.reset()

      val weightedProduct = rows.gramianProduct(direction, Some(weights))

      assert(weightedProduct ~== Vectors.dense(276.0, 24.0, 432.0) absTol 1e-12)
      assert(reads.value == 2L)
      assert(direction.value.values.sameElements(Array(1.0, -2.0, 3.0)))
    } finally direction.destroy()
  }

  test("Gramian products of empty inputs return zero with zero or eight partitions") {
    val direction = sc.broadcast(new DenseVector(Array(1.0, 2.0, 3.0)))
    val emptyInputs: Seq[DMatrix] = Seq(
      sc.emptyRDD[Vector], sc.parallelize(Seq.empty[Vector], 8))
    try {
      emptyInputs.foreach { rows =>
        val product = rows.gramianProduct(direction)

        assert(product.values.sameElements(Array(0.0, 0.0, 0.0)))
      }
    } finally direction.destroy()
  }

  test("packed Gramian retains dense and sparse outer products at each aggregation depth") {
    val matrix: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.sparse(3, Seq(0 -> 4.0, 2 -> 6.0))), 8)
    val aggregationDepths = Seq(1, 2, 3)
    // Packed upper triangle of the sum of the two outer products.
    val expected = Array(17.0, 2.0, 4.0, 27.0, 6.0, 45.0)

    aggregationDepths.foreach { depth =>
      val gramian = matrix.gramianMatrix(3, depth).toArray

      assert(gramian.sameElements(expected), s"aggregation depth $depth")
    }
  }

  test("packed Gramian of empty inputs returns six zeros with zero or eight partitions") {
    val emptyInputs: Seq[DMatrix] = Seq(
      sc.emptyRDD[Vector], sc.parallelize(Seq.empty[Vector], 8))

    emptyInputs.foreach { matrix =>
      val gramian = matrix.gramianMatrix(3).toArray

      assert(gramian.sameElements(Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    }
  }

  test("transpose exchanges rows and columns while preserving the original matrix") {
    val originalRows = Seq(
      Vectors.dense(-1.0, -1.0, 1.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 1.0))
    val matrix: DMatrix = sc.parallelize(originalRows)

    val transposedRows = matrix.t.collect().toSeq
    val rowsAfterTranspose = matrix.collect().toSeq

    assert(transposedRows == Seq(
      Vectors.dense(-1.0, 1.0), Vectors.dense(-1.0, 0.0),
      Vectors.dense(1.0, 0.0), Vectors.dense(0.0, 1.0)))
    assert(rowsAfterTranspose == originalRows)
  }

  test("Gramian diagonals include every row and apply each supplied row weight") {
    val matrix: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(4.0, 5.0, 6.0),
      Vectors.dense(7.0, 8.0, 9.0), Vectors.dense(1.0, 0.0, -1.0)), 2)
    val weights: DVector = sc.parallelize(Seq(0.5, 2.0, 1.0, 3.0), 2).glom.map(new DenseVector(_))

    val diagonal = matrix.gramianDiagonal()
    val weightedDiagonal = matrix.gramianDiagonal(Some(weights))

    assert(Vectors.dense(diagonal.toArray) ~== Vectors.dense(67.0, 93.0, 127.0) absTol 1e-12)
    assert(Vectors.dense(weightedDiagonal.toArray) ~== Vectors.dense(84.5, 116.0, 160.5) absTol 1e-12)
  }

  test("Gramian diagonals include implicit zeros from sparse rows") {
    val matrix: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0), Vectors.sparse(3, Seq(0 -> 4.0, 2 -> 6.0))), 1)
    val weights: DVector = sc.parallelize(Seq(2.0, 3.0), 1).glom.map(new DenseVector(_))

    val diagonal = matrix.gramianDiagonal()
    val weightedDiagonal = matrix.gramianDiagonal(Some(weights))

    assert(Vectors.dense(diagonal.toArray) ~== Vectors.dense(17.0, 4.0, 45.0) absTol 1e-12)
    assert(Vectors.dense(weightedDiagonal.toArray) ~== Vectors.dense(50.0, 8.0, 126.0) absTol 1e-12)
  }

  test("selected Gramian columns retain requested column-major order with sparse weighted rows") {
    val matrix: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0), Vectors.sparse(3, Seq(0 -> 4.0, 2 -> 6.0))), 2)
    val weights: DVector = sc.parallelize(Seq(2.0, 3.0), 2).glom.map(new DenseVector(_))
    val selectedColumns = Array(2, 0)

    val columns = matrix.gramianColumns(selectedColumns)
    val weightedColumns = matrix.gramianColumns(selectedColumns, Some(weights))

    assert(Vectors.dense(columns) ~==
      Vectors.dense(27.0, 6.0, 45.0, 17.0, 2.0, 27.0) absTol 1e-12)
    assert(Vectors.dense(weightedColumns) ~==
      Vectors.dense(78.0, 12.0, 126.0, 50.0, 4.0, 78.0) absTol 1e-12)
  }

  test("gramianColumns preserves adjacent and nonadjacent repeated requests") {
    val matrix: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.sparse(3, Seq(0 -> 4.0, 2 -> 6.0))), 2)
    val weights: DVector = sc.parallelize(Array(2.0, 3.0), 2).glom.map(new DenseVector(_))
    val requests = Seq(
      (Array(0, 0), Array(17.0, 2.0, 27.0, 17.0, 2.0, 27.0),
        Array(50.0, 4.0, 78.0, 50.0, 4.0, 78.0)),
      (Array(2, 0, 2), Array(27.0, 6.0, 45.0, 17.0, 2.0, 27.0, 27.0, 6.0, 45.0),
        Array(78.0, 12.0, 126.0, 50.0, 4.0, 78.0, 78.0, 12.0, 126.0)))

    requests.foreach { case (indices, expected, expectedWeighted) =>
      val result = matrix.gramianColumns(indices)
      val weighted = matrix.gramianColumns(indices, Some(weights))

      assert(result.sameElements(expected))
      assert(weighted.sameElements(expectedWeighted))
    }
  }

  test("gramianColumns preserves repeated columns absent from sparse rows") {
    val matrix: DMatrix = sc.parallelize(Seq(Vectors.sparse(3, Seq(0 -> 2.0, 2 -> 3.0))), 1)

    val result = matrix.gramianColumns(Array(1, 0, 1))

    assert(result.sameElements(Array(0.0, 0.0, 0.0, 4.0, 0.0, 6.0, 0.0, 0.0, 0.0)))
  }
}
