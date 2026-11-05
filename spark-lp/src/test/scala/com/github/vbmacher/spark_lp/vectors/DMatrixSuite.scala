package com.github.vbmacher.spark_lp.vectors

import com.github.vbmacher.spark_lp.TestingUtils._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import dmatrix.implicits._
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.scalatest.funsuite.AnyFunSuite

class DMatrixSuite extends AnyFunSuite with DataFrameSuiteBase {

  test("fused Gramian products read each matrix row once and handle mixed rows and empty partitions") {
    val reads = sc.longAccumulator("gramian-product-rows")
    val rows: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.sparse(3, Seq(0 -> 4.0, 2 -> 6.0))), 8).map { row =>
      reads.add(1L)
      row
    }
    val weights: DVector = sc.parallelize(Seq(2.0, 3.0), 8).glom().map(new DenseVector(_))
    val p = sc.broadcast(new DenseVector(Array(1.0, -2.0, 3.0)))
    try {
      val ops = new DMatrixOps(rows)
      assert(ops.gramianProduct(p) ~== Vectors.dense(94.0, 12.0, 150.0) absTol 1e-12)
      assert(reads.value == 2L)
      reads.reset()
      assert(ops.gramianProduct(p, Some(weights)) ~== Vectors.dense(276.0, 24.0, 432.0) absTol 1e-12)
      assert(reads.value == 2L)
      assert(p.value.values.sameElements(Array(1.0, -2.0, 3.0)))
    } finally p.destroy()
  }

  test("fused Gramian product of empty input is zero with or without partitions") {
    val p = sc.broadcast(new DenseVector(Array(1.0, 2.0, 3.0)))
    try {
      Seq(sc.emptyRDD[org.apache.spark.mllib.linalg.Vector],
        sc.parallelize(Seq.empty[org.apache.spark.mllib.linalg.Vector], 8)).foreach { rows =>
        assert(rows.gramianProduct(p).values.sameElements(Array(0.0, 0.0, 0.0)))
      }
    } finally p.destroy()
  }

  test("packed Gramian combines dense and sparse rows with empty partitions") {
    val matrix: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.sparse(3, Seq(0 -> 4.0, 2 -> 6.0))), 8)
    // Packed upper triangle of the explicit sum of the two outer products.
    val expected = Array(17.0, 2.0, 4.0, 27.0, 6.0, 45.0)
    Seq(1, 2, 3).foreach { depth =>
      assert(matrix.gramianMatrix(3, depth).toArray.sameElements(expected))
    }
  }

  test("packed Gramian of empty input is zero with or without partitions") {
    val empty: DMatrix = sc.emptyRDD[org.apache.spark.mllib.linalg.Vector]
    val partitions: DMatrix = sc.parallelize(Seq.empty[org.apache.spark.mllib.linalg.Vector], 8)
    Seq(empty, partitions).foreach { matrix =>
      assert(matrix.gramianMatrix(3).toArray.sameElements(Array.fill(6)(0.0)))
    }
  }

  test("transpose works") {
    val matrix = Array(
      Array(-1.0, -1.0, 1.0, 0.0),
      Array(1.0, 0.0, 0.0, 1.0))

    val matrixT = matrix.transpose

    val dMatrix = sc.parallelize(matrix).map(Vectors.dense)
    val dMatrixT = dMatrix.t

    assert(matrix.map(Vectors.dense) === dMatrix.collect())
    assert(matrixT.map(Vectors.dense) === dMatrixT.collect())
  }

  test("gramianDiagonal computes diag(A^T A) and diag(A^T diag(w) A)") {
    val matrix = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
      Array(7.0, 8.0, 9.0),
      Array(1.0, 0.0, -1.0))
    val w = Array(0.5, 2.0, 1.0, 3.0)

    val dMatrix: DMatrix = sc.parallelize(matrix, 2).map(Vectors.dense)
    val dW: DVector = sc.parallelize(w, 2).glom.map(new DenseVector(_))

    val unweighted = dMatrix.gramianDiagonal()
    assert(Vectors.dense(unweighted.toArray) ~== Vectors.dense(67.0, 93.0, 127.0) absTol 1e-12)

    val weighted = dMatrix.gramianDiagonal(Some(dW))
    assert(Vectors.dense(weighted.toArray) ~== Vectors.dense(84.5, 116.0, 160.5) absTol 1e-12)
  }

  test("gramianDiagonal is sparse safe") {
    val dMatrix: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.sparse(3, Seq((0, 4.0), (2, 6.0)))), 1)
    val dW: DVector = sc.parallelize(Array(2.0, 3.0), 1).glom.map(new DenseVector(_))

    assert(Vectors.dense(dMatrix.gramianDiagonal().toArray) ~== Vectors.dense(17.0, 4.0, 45.0) absTol 1e-12)
    assert(Vectors.dense(dMatrix.gramianDiagonal(Some(dW)).toArray) ~== Vectors.dense(50.0, 8.0, 126.0) absTol 1e-12)
  }

  test("gramianColumns computes sparse weighted selected columns in column-major order") {
    val dMatrix: DMatrix = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.sparse(3, Seq((0, 4.0), (2, 6.0)))), 2)
    val dW: DVector = sc.parallelize(Array(2.0, 3.0), 2).glom.map(new DenseVector(_))
    val indices = Array(2, 0)

    assert(Vectors.dense(dMatrix.gramianColumns(indices)) ~==
      Vectors.dense(27.0, 6.0, 45.0, 17.0, 2.0, 27.0) absTol 1e-12)
    assert(Vectors.dense(dMatrix.gramianColumns(indices, Some(dW))) ~==
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
