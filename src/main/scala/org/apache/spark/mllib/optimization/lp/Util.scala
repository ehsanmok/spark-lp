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
  */

package org.apache.spark.mllib.optimization.lp

import breeze.linalg.{DenseMatrix => BDM}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.apache.spark.mllib.linalg.{Matrices, Matrix}
import org.netlib.util.intW

private[lp] object Util {

  /**
    * Transform the breeze dense matrix into packed LAPACK acceptable column-major format.
    *
    * @param mat The input breeze dense vector.
    * @return The packed LAPACK column-major format.
    */
  def toUpperTriangularArray(mat: BDM[Double]): Array[Double] = {

    val m = mat.rows // numRows
    val k = if (m % 2 == 0) (m / 2) * (m + 1) else ((m + 1) / 2) * m // be careful about Int overflow
    val result = new Array[Double](k)
    var j = 0
    while (j < m) {
      var i = 0
      while (i <= j) {
        result(i + j * (j + 1) / 2) = mat(i, j)
        i += 1
      }
      j += 1
    }
    result
  }

  /**
    * Compute the inverse of a symmetric positive definite matrix "inplace" using LAPACK routines.
    *
    * @param B The column-major LAPACK format for symmetric matrices.
    * @param m The number of columns.
    */
  def posSymDefInv(B: Array[Double], m: Int): Unit = {
    val info1 = new intW(0)
    lapack.dpptrf("U", m, B, info1) // cholesky decomposition of sym pos def packed format
    val code1 = info1.`val`
    assert(code1 == 0, s"lapack. returned $code1.")
    val info2 = new intW(0)
    lapack.dpptri("U", m, B, info2) // input chol dec above packed output packed inv
    val code2 = info2.`val`
    assert(code2 == 0, s"lapack. returned $code2.")
  }

  /**
    * Fills a full square matrix from its upper triangular part.
    *
    * @param U The upper triangular part.
    * @param n The number of rows or columns.
    * @return The filled matrix for its other half.
    */
  def triuToFull(U: Array[Double], n: Int): Matrix = {
    val G = new BDM[Double](n, n)

    var row = 0
    var col = 0
    var idx = 0
    var value = 0.0
    while (col < n) {
      row = 0
      while (row < col) {
        value = U(idx)
        G(row, col) = value
        G(col, row) = value
        idx += 1
        row += 1
      }
      G(col, col) = U(idx)
      idx += 1
      col +=1
    }

    Matrices.dense(n, n, G.data)
  }
}
