package com.github.vbmacher.spark_lp.vectors

import breeze.linalg.{DenseVector => BDV}
import com.github.vbmacher.spark_lp.collections.implicits.IteratorOps
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.wrappers.BLAS
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}

object dmatrix {

  object functions extends LazyLogging  {
    /**
      * Computes the packed upper triangle of `matrix^T matrix` on the driver.
      *
      * @param ncol number of columns in every matrix row; at most 65,535.
      * @param depth depth of Spark's tree aggregation; it affects the reduction plan, not the result.
      * @return packed upper triangle in column-major BLAS order.
      */
    def gramianMatrix(matrix: DMatrix, ncol: Int, depth: Int = 2): BDV[Double] = {

      checkNumColumns(ncol)
      // Computes n*(n+1)/2, avoiding overflow in the multiplication.
      // This succeeds when n <= 65535, which is checked above
      val nt =
        if (ncol % 2 == 0) (ncol / 2) * (ncol + 1)
        else ncol * ((ncol + 1) / 2)

      // Allocate the packed accumulator on executors, not in the serialized task closure.
      // A dense zero value otherwise sends O(ncol^2) bytes before any rows are processed.
      val GU = matrix.treeAggregate[BDV[Double]](null)(
        seqOp = (U, v) => {
          val accumulator = if (U == null) new BDV[Double](nt) else U
          BLAS.spr(1.0, v, accumulator.data)
          accumulator
        }, combOp = (U1, U2) => {
          if (U1 == null) U2
          else if (U2 == null) U1
          else U1 += U2
        }, depth)
      // An entirely empty matrix has an all-zero Gramian, including with zero partitions.
      if (GU == null) new BDV[Double](nt) else GU // BLAS packed columnwise format
    }

    /** Rejects dimensions that cannot be represented by the packed-triangle array. */
    private def checkNumColumns(cols: Int): Unit = {
      if (cols > 65535) {
        throw new IllegalArgumentException(s"Argument with more than 65535 cols: $cols")
      }
      if (cols > 10000) {
        val memMB = (cols.toLong * cols) / 125000
        logger.warn(s"$cols columns will require at least $memMB megabytes of memory!")
      }
    }
  }

  object implicits {

    implicit class DMatrixOps(matrix: DMatrix) extends LazyLogging {

      private lazy val columns = matrix.first().size

      /**
        * Materializes the distributed transpose.
        *
        * Each output row is one input column, ordered by the original row index.
        */
      lazy val t: DMatrix = {
        // Convert each vector (row) into an indexed sequence of tuples (colIndex, value)
        val indexedRows = matrix.zipWithIndex.flatMap {
          case (vector, rowIndex) => vector.toArray.zipWithIndex.map {
            case (value, colIndex) => (colIndex, (rowIndex, value))
          }
        }

        // Group by column index and sort by row index to form the transposed rows
        val transposed = indexedRows.groupByKey().sortByKey().map {
          case (_, rowValues) =>
            Vectors.dense(rowValues.toSeq.sortBy(_._1).map(_._2).toArray)
        }

        transposed
      }

      /**
        * Computes the packed upper triangle of `matrix^T matrix` on the driver.
        *
        * @param ncol number of columns in every matrix row; at most 65,535.
        * @param depth depth of Spark's tree aggregation; it affects the reduction plan, not the result.
        * @return packed upper triangle in column-major BLAS order.
        */
      def gramianMatrix(ncol: Int, depth: Int = 2): BDV[Double] = {
        functions.gramianMatrix(matrix, ncol, depth)
      }

      /**
        * Computes `matrix^T x` and returns the result on the driver.
        *
        * Corresponding partitions must align: a matrix partition with `k` rows must match a vector
        * partition containing `k` elements.
        *
        * @param x distributed vector paired with the matrix rows.
        * @param depth depth of Spark's tree aggregation.
        */
      def adjointProduct(x: DVector, depth: Int = 2): DenseVector = {
        val n = columns
        // Merge two partition-sized accumulators in place: sum1 += sum2. Captures nothing, so it is
        // safe to reuse across the partition aggregate and the tree reduction closures.
        val add = (sum1: DenseVector, sum2: DenseVector) => { BLAS.axpy(1.0, sum2, sum1); sum1 }
        matrix.zipPartitions(x)((matrixPartition, xPartition) =>
          Iterator.single(
            matrixPartition
              .checkedZip(xPartition.next.values.toIterator) // ignoring more rows
              .aggregate(Vectors.zeros(n).toDense)( // NOTE A DenseVector result is assumed here (not sparse safe).
                seqop = {
                  case (sum, (matrix_i, x_i)) =>
                    // Multiply an element of x by its corresponding matrix row, and add to the accumulation sum vector.
                    BLAS.axpy(x_i, matrix_i, sum)
                    sum
                },
                combop = add
              ))
        ).treeAggregate(Vectors.zeros(n).toDense)(seqOp = add, combOp = add, depth)
      }

      /** Returns `matrix * x`, reading the broadcast vector on each executor. */
      def product(x: Broadcast[DenseVector]): DVector = rowDots(row => BLAS.dot(row, x.value))

      /** Returns `matrix * x`; Spark serializes `x` with each task closure. */
      def product(x: DenseVector): DVector = rowDots(row => BLAS.dot(row, x))

      // Dot each matrix row with a per-row supplied vector. A broadcast argument stays a broadcast:
      // its `value` is dereferenced inside `dot`, on the executor, not captured in the task closure.
      // NOTE A DenseVector result is assumed here (not sparse safe).
      private def rowDots(dot: Vector => Double): DVector =
        matrix.mapPartitions(partitionRows =>
          Iterator.single(new DenseVector(partitionRows.map(dot).toArray)))

      /**
        * Computes `matrix^T diag(w) matrix x` without constructing the Gram matrix.
        *
        * A supplied `w` must align with the matrix rows as described by [[adjointProduct]]. Only one
        * vector with the matrix column count is reduced to the driver.
        */
      def gramianProduct(x: Broadcast[DenseVector], w: Option[DVector] = None,
        depth: Int = 2): DenseVector = {
        val n = x.value.size
        val perPartition = w match {
          case Some(weights) =>
            matrix.zipPartitions(weights)((rows, weightPartition) => {
              val p = x.value
              val sum = Vectors.zeros(n).toDense
              rows.checkedZip(weightPartition.next().values.toIterator).foreach { case (row, wi) =>
                BLAS.axpy(wi * BLAS.dot(row, p), row, sum)
              }
              Iterator.single(sum)
            })
          case None =>
            matrix.mapPartitions(rows => {
              val p = x.value
              val sum = Vectors.zeros(n).toDense
              rows.foreach(row => BLAS.axpy(BLAS.dot(row, p), row, sum))
              Iterator.single(sum)
            })
        }
        val merge = (left: DenseVector, right: DenseVector) =>
          if (left == null) right else if (right == null) left else { BLAS.axpy(1.0, right, left); left }
        val result = perPartition.treeAggregate[DenseVector](null)(merge, merge, depth)
        if (result == null) Vectors.zeros(n).toDense else result
      }

      /**
        * Reduces one `size`-element accumulator per partition to the driver. A supplied weight vector
        * must align with the matrix rows as described by [[adjointProduct]].
        */
      private def weightedRowAccumulate(w: Option[DVector], size: Int, depth: Int)(
        add: (Array[Double], Vector, Double) => Unit): Array[Double] = {
        val perPartition = w match {
          case Some(weights) =>
            matrix.zipPartitions(weights)((rows, wPartition) => {
              val acc = new Array[Double](size)
              rows.checkedZip(wPartition.next().values.toIterator).foreach { case (row, wi) => add(acc, row, wi) }
              Iterator.single(acc)
            })
          case None =>
            matrix.mapPartitions(rows => {
              val acc = new Array[Double](size)
              rows.foreach(row => add(acc, row, 1.0))
              Iterator.single(acc)
            })
        }
        perPartition.treeAggregate(new Array[Double](size))(
          seqOp = (a, b) => { new BDV(a) += new BDV(b); a },
          combOp = (a, b) => { new BDV(a) += new BDV(b); a }, depth)
      }

      /**
        * Computes the diagonal of `matrix^T diag(w) matrix` in one distributed pass.
        *
        * A supplied `w` must align with the matrix rows as described by [[adjointProduct]]. The full
        * Gram matrix is never constructed, and driver memory is linear in the column count.
        *
        * @param w optional row weights; `None` uses a weight of one for every row.
        * @param depth depth of Spark's tree aggregation.
        */
      def gramianDiagonal(w: Option[DVector] = None, depth: Int = 2): DenseVector =
        new DenseVector(weightedRowAccumulate(w, columns, depth) { (acc, row, weight) =>
          row.foreachActive((j, v) => acc(j) += weight * v * v)
        })

      /**
        * Computes selected columns of `matrix^T diag(w) matrix` in one distributed pass.
        *
        * The result is column-major: for `n` matrix columns, result entry `s * n + i` is row `i` of
        * the `s`th requested column. A supplied `w` must align with the matrix rows as described by
        * [[adjointProduct]]. Driver and per-partition memory are `O(n * indices.length)`.
        *
        * @param indices Gram-matrix column indices, in requested output order; repetitions are preserved.
        * @param w optional row weights; `None` uses a weight of one for every row.
        * @param depth depth of Spark's tree aggregation.
        */
      def gramianColumns(indices: Array[Int], w: Option[DVector] = None, depth: Int = 2): Array[Double] = {
        val n = columns
        val k = indices.length
        val slots = Array.fill(n)(-1)
        indices.zipWithIndex.foreach { case (column, s) => slots(column) = s }

        // Each row contributes the rank-1 update `weight * row * row^T`; only the columns with a
        // slot are accumulated. foreachActive keeps the pass sparse-safe.
        def accumulate(acc: Array[Double], row: Vector, weight: Double): Unit = {
          row.foreachActive { (i, v) =>
            val s = slots(i)
            if (s >= 0 && v != 0.0) {
              val coefficient = weight * v
              val offset = s * n
              row.foreachActive((j, u) => acc(offset + j) += coefficient * u)
            }
          }
        }

        val result = weightedRowAccumulate(w, n * k, depth)(accumulate)
        indices.zipWithIndex.foreach { case (column, s) =>
          val source = slots(column)
          if (source != s) Array.copy(result, source * n, result, s * n, n)
        }
        result
      }
    }
  }
}
