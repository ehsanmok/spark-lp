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

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.TestingUtils._

import org.scalatest.FunSuite

import breeze.linalg.{ DenseMatrix => BDM }

class UtilSuite extends FunSuite {

  test(" toUpperTriangularArray is implemented properly") {
    val A: BDM[Double] = new BDM[Double](3, 3,
      Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
    assert(Util.toUpperTriangularArray(A).deep ==  Array(1.0, 4.0, 5.0, 7.0, 8.0, 9.0).deep,
    "Arrays are not equal!")
  }

  test("posSymDefInv is implemented properly") {
    val A = Array(5.0, 8.0, 13.0) // packed columnwise format sym pos def mat
    Util.posSymDefInv(A, 2)
    assert(Vectors.dense(A) ~== Vectors.dense(Array(13.0, -8.0, 5.0)) absTol 1e-8 )
  }
}
