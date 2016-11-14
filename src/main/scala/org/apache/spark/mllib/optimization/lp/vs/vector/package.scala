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

import org.apache.spark.mllib.linalg.{BLAS, DenseVector}
import org.apache.spark.mllib.optimization.lp.VectorSpace

package object vector {

  /** A VectorSpace implementation for DenseVectors in local memory. */
  implicit object DenseVectorSpace extends VectorSpace[DenseVector] {

    override def combine(alpha: Double,
                         a: DenseVector,
                         beta: Double,
                         b: DenseVector): DenseVector = {
      val ret = a.copy
      BLAS.scal(alpha, ret)
      BLAS.axpy(beta, b, ret)
      ret
    }

    override def dot(a: DenseVector, b: DenseVector): Double = BLAS.dot(a, b)

    override def entrywiseProd(a: DenseVector, b: DenseVector): DenseVector = {
      val c = a.values.zip(b.values).map { case (i: Double, j: Double) => i * j }
      new DenseVector(c)
    }

    override def entrywiseNegDiv(a: DenseVector, b: DenseVector): DenseVector = {
      val c = a.values.zip(b.values).map {
        case (ai, bi) if bi < 0 => ai / Math.max(Math.abs(bi), 1e-15)
        case (_, bi) if bi >= 0 => Double.PositiveInfinity // Make Infinity value to be neglected in min
      }
      new DenseVector(c)
    }

    override def sum(a: DenseVector): Double = a.values.sum

    override def max(a: DenseVector): Double = a.values.max

    override def min(a: DenseVector): Double = a.values.min

  }
}
