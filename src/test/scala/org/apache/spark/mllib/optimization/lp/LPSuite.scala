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

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.mllib.optimization.lp.VectorSpace._
import org.apache.spark.mllib.optimization.lp.vs.dvector.DVectorSpace
import org.apache.spark.mllib.optimization.lp.vs.vector.DenseVectorSpace

class LPSuite extends FunSuite with MLlibTestSparkContext {

  val numPartitions = 2
  val cArray = Array(2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0)
  val BArray = Array(
    Array(12.0, 16.0, 30.0, 1.0, 0.0),
    Array(24.0, 16.0, 12.0, 0.0, 1.0),
    Array(-1.0, 0.0, 0.0, 0.0, 0.0),
    Array(0.0, -1.0, 0.0, 0.0, 0.0),
    Array(0.0, 0.0, -1.0, 0.0, 0.0),
    Array(0.0, 0.0, 0.0, 1.0, 0.0),
    Array(0.0, 0.0, 0.0, 0.0, 1.0))
  val bArray = Array(120.0, 120.0, 120.0, 15.0, 15.0)

  lazy val c = sc.parallelize(cArray, numPartitions).glom.map(new DenseVector(_))
  lazy val rows = sc.parallelize(BArray, numPartitions).map(Vectors.dense(_))
  lazy val b = new DenseVector(bArray)

  test("LP solve is implemented properly") {
    val (v, x) = LP.solve(c, rows, b, sc=sc)
    // solution obtained from scipy.optimize.linprog and octave glgk lpsolver with fun_val = 12.083
    val expectedSol = Vectors.dense(
      Array(1.66666667, 5.83333333, 40.0, 0.0, 0.0, 13.33333333, 9.16666667))
    val xx = Vectors.dense(x.flatMap(_.toArray).collect())
    println(s"$xx")
    println("optimal min value: " + v)
    assert(xx ~== expectedSol absTol 1e-6, "LP.solve x should return the correct answer.")

  }

}