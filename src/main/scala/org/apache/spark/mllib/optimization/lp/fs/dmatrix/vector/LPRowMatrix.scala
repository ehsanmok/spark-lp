/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
  * @author Ehsan Mohyedin Kermani: ehsanmo1367@gmail.com
  * Appropriate modifications of
  * @see[[https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/linalg/distributed/RowMatrix.scala]]
  */

package org.apache.spark.mllib.optimization.lp

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

import com.github.fommil.netlib.{BLAS => NetlibBLAS}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{BLAS, Matrices, Matrix}
import org.apache.spark.mllib.optimization.lp.VectorSpace._
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.apache.spark.storage.StorageLevel

/**
  * Represents a row-oriented distributed Matrix with no meaningful row indices.
  *
  * @param rows The rows stored as an RDD[Vector]
  * @param nRows The number of rows. A non-positive value means unknown, and then the number of rows will
  *              be determined by the number of records in the RDD `rows`.
  * @param nCols The number of columns. A non-positive value means unknown, and then the number of
  *              columns will be determined by the size of the first row.
  */
class LPRowMatrix (val rows: DMatrix,
                   val nRows: Long,
                   val nCols: Int) extends Serializable with Logging {


  if (rows.getStorageLevel == StorageLevel.NONE) {
    rows.cache()
  }

  /**
    * Computes the Gramian matrix `A^T A`. Note that this cannot be computed on matrices with
    * more than 65535 columns.
    */
  def computeGramianMatrixColumn(ncol: Int, depth: Int = 2): BDV[Double] = {

    checkNumColumns(ncol)
    // Computes n*(n+1)/2, avoiding overflow in the multiplication.
    // This succeeds when n <= 65535, which is checked above
    val nt = if (ncol % 2 == 0) ((ncol / 2) * (ncol + 1)) else (ncol * ((ncol + 1) / 2))

    // Compute the upper triangular part of the gram matrix.
    val GU = rows.treeAggregate(new BDV[Double](nt))(
      seqOp = (U, v) => {
        BLAS.spr(1.0, v, U.data)
        //NativeBLAS.dspr("U", ncol, 1.0, v, 1, U) //symmetric rk 1 update included in BLAS netlib-java
        U
      }, combOp = (U1, U2) => U1 += U2, depth)
    GU // column major == BLAS packed columnwise format
  }

  /**
    * Check if the number of columns exceed 65535 to avoid Array overflow
    *
    * @param cols The number of columns
    */
  private def checkNumColumns(cols: Int): Unit = {
    if (cols > 65535) {
      throw new IllegalArgumentException(s"Argument with more than 65535 cols: $cols")
    }
    if (cols > 10000) {
      val memMB = (cols.toLong * cols) / 125000
      logWarning(s"$cols columns will require at least $memMB megabytes of memory!")
    }
  }
}