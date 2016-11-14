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

package org.apache.spark.mllib.optimization.lp.fs.dvector.dmatrix

import org.apache.spark.mllib.linalg.{BLAS, Vector}
import org.apache.spark.mllib.optimization.lp.CheckedIteratorFunctions._
import org.apache.spark.mllib.optimization.lp.LinearOperator
import org.apache.spark.mllib.optimization.lp.VectorSpace._
import org.apache.spark.storage.StorageLevel

/**
  * Compute the product of a DVector to a DMatrix where each element of DVector is multiplied to
  * the corresponding row to produce a DMatrix. This is for optimizing the product of diagonal
  * DMatrix to a DMatrix.
  *
  * @param dvector The DVector representing a diagonal matrix.
  */
class SpLinopMatrix(@transient private val dvector: DVector)
  extends LinearOperator[DMatrix, DMatrix] with Serializable {

  if (dvector.getStorageLevel == StorageLevel.NONE) {
    dvector.cache()
  }

  /**
    * Apply the multiplication.
    *
    * @param mat The DMatrix for multiplication.
    * @return The result of applying the operator on x.
    */
  override def apply(mat: DMatrix): DMatrix = {
    dvector.zipPartitions(mat)((vectorPartition, matPartition) =>
      vectorPartition.next().values.toIterator.checkedZip(matPartition.toIterator).map {
          case (a: Double, x: Vector) =>
            val xc = x.copy
              BLAS.scal(a, xc)
            xc
        }
      )
  }
}
