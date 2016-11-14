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

import org.scalatest.FunSuite

import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.optimization.lp.vs.dvector.DVectorSpace
import org.apache.spark.mllib.optimization.lp.VectorSpace._
import org.apache.spark.mllib.optimization.lp.fs.dvector.dmatrix._

class SpLinopMatrixSuite extends FunSuite with MLlibTestSparkContext {

  test("SpLinopMatrix.apply is implemented properly") {

    val matrix: DMatrix = sc.parallelize(Array(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.dense(4.0, 5.0, 6.0)),
      2)

    val vector: DVector = sc.parallelize(Array(2.0, 3.0), 2).glom.map(new DenseVector(_))

    val expectApply: DMatrix = sc.parallelize(Array(
      Vectors.dense(2.0 * 1.0, 2.0 * 2.0, 2.0 * 3.0),
      Vectors.dense(3.0 * 4.0, 3.0 * 5.0, 3.0 * 6.0)),
      2)
    assert((new SpLinopMatrix(vector))(matrix).collect().deep == expectApply.collect().deep, // or sameElements
      "SpLinopMatrix.apply should return the correct result.")
  }
}