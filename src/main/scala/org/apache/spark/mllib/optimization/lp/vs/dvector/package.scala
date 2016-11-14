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
  * @author Aaron Staple, Ehsan Mohyedin Kermani: ehsanmo1367@gmail.com
  */

package org.apache.spark.mllib.optimization.lp.vs

import org.apache.spark.mllib.linalg.{DenseVector, BLAS, Vector}
import org.apache.spark.mllib.optimization.lp.DVectorFunctions._
import org.apache.spark.mllib.optimization.lp.VectorSpace
import org.apache.spark.mllib.optimization.lp.VectorSpace._
import org.apache.spark.mllib.optimization.lp.vs.vector.DenseVectorSpace
import org.apache.spark.storage.StorageLevel

package object dvector {

  /** A VectorSpace implementation for (distributed) DVector vectors. */
  implicit object DVectorSpace extends VectorSpace[DVector] {

    override def combine(alpha: Double, a: DVector, beta: Double, b: DVector): DVector =
      if (alpha == 1.0 && beta == 1.0) {
        a.zip(b).map {
          case (aPart, bPart) => {
            BLAS.axpy(1.0, aPart, bPart) // bPart += aPart
            bPart
          }
        }
      } else {
        a.zip(b).map {
          case (aPart, bPart) =>
            // NOTE A DenseVector result is assumed here (not sparse safe).
            DenseVectorSpace.combine(alpha, aPart, beta, bPart).toDense
        }
      }

    override def dot(a: DVector, b: DVector): Double = a.dot(b)

    override def entrywiseProd(a: DVector, b: DVector): DVector = {
      a.zip(b).map {
        case (aPart, bPart) =>
          DenseVectorSpace.entrywiseProd(aPart, bPart).toDense
      }
    }

    override def entrywiseNegDiv(a: DVector, b: DVector): DVector = {
      a.zip(b).map {
        case (aPart, bPart) =>
            DenseVectorSpace.entrywiseNegDiv(aPart, bPart)
      }
    }

    override def sum(a: DVector): Double = a.aggregate(0.0)(
      seqOp = (acc: Double, v: DenseVector) => acc + v.values.sum,
      combOp = (acc1: Double, acc2: Double) => acc1 + acc2
    )

    override def min(a: DVector): Double = a.aggregate(Double.PositiveInfinity)(
      (mi, x) => Math.min(mi, x.values.min), Math.min
    )

    override def max(a: DVector): Double = a.aggregate(Double.NegativeInfinity)(
      (ma, x) => Math.max(ma, x.values.max), Math.max
    )


    override def cache(a: DVector): Unit =
      if (a.getStorageLevel == StorageLevel.NONE) {
        a.cache()
      }
  }
}
