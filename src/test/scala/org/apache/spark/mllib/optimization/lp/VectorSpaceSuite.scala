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

import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.optimization.lp.DVectorFunctions._
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.mllib.optimization.lp.vs.dvector.DVectorSpace
import org.apache.spark.mllib.optimization.lp.vs.vector.DenseVectorSpace

class VectorSpaceSuite extends FunSuite with MLlibTestSparkContext {
  // taken from first iteration
  test("DenseVectorSpace.combine is implemented properly") {
    val alpha = -1.0
    val a = new DenseVector(Array(
      1037.8830194476832,919.2678172250901,1215.8058227815707,59.30760111129604,59.307601111296094
    ))
    val beta = 1.0
    val b = new DenseVector(Array(
      130.74622637382816,109.16609581526643,124.20404493660249,10.918025554611186,11.577626670204133
    ))
    val expectedCombination = Vectors.dense(
      -907.1367930738562, -810.1017214098238, -1091.6017778449684, -48.38957555668489, -47.729974441092
    )
    assert(DenseVectorSpace.combine(alpha, a, beta, b) ~= expectedCombination relTol 1e-6,
      "DenseVectorSpace.combine should return the correct result.")
  }

  test("DenseVectorSpace.dot is implemented properly") {
    val a = new DenseVector(Array(2.0, 3.0))
    val b = new DenseVector(Array(5.0, 6.0))
    val expectedDot = 2.0 * 5.0 + 3.0 * 6.0
    assert(DenseVectorSpace.dot(a, b) == expectedDot,
      "DenseVectorSpace.dot should return the correct result.")
  }

  test("DenseVectorSpace.entrywiseProd is implemented properly") {
    val a = new DenseVector(Array(2.0, 3.0))
    val b = new DenseVector(Array(5.0, 6.0))
    val expectedEntrywiseProd = new DenseVector(Array(2.0 * 5.0, 3.0 * 6.0))
    assert(DenseVectorSpace.entrywiseProd(a, b) == expectedEntrywiseProd,
      "DenseVectorSpace.entrywiseProd should return the correct result.")
  }

  test("DenseVectorSpace.entrywiseNegDiv is implemented properly") {
    val a = new DenseVector(Array(2,0, 3.0))
    val b = new DenseVector(Array(-1.0, 1.0))
    val expectedEntrywiseNegDiv = new DenseVector(Array(2.0 / math.abs(-1.0), Double.PositiveInfinity ))
    assert(DenseVectorSpace.entrywiseNegDiv(a, b) == expectedEntrywiseNegDiv,
      "DenseVectorSpace.entrywiseNegDiv should return the correct result.")
  }

  test("DenseVectorSpace.sum is implemented properly") {
    val a = new DenseVector(Array(1.0, 2,0, 3.0))
    val expectedSum = 1.0 + 2.0 + 3.0
    assert(DenseVectorSpace.sum(a) == expectedSum,
      "DenseVectorSpace.sum should return the correct result.")
  }

  test("DenseVectorSpace.max is implemented properly") {
    val a = new DenseVector(Array(1.0, 2,0, 3.0))
    val expectedMax = 3.0
    assert(DenseVectorSpace.max(a) == expectedMax,
      "DenseVectorSpace.max should return the correct result.")
  }

  test("DenseVectorSpace.min is implemented properly") {
    val a = new DenseVector(Array(1.0, 2.0, 3.0))
    val expectedMin = 1.0
    assert(DenseVectorSpace.min(a) == expectedMin,
      "DenseVectorSpace.min should return the correct result.")
  }

  test("DVectorSpace.combine is implemented properly") {
    val alpha = 1.1
    val a = sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))), 2)
    val beta = 4.0
    val b = sc.parallelize(Array(new DenseVector(Array(5.0, 6.0)), new DenseVector(Array(7.0))), 2)
    val combination = DVectorSpace.combine(alpha, a, beta, b)
    val expectedCombination =
      Vectors.dense(1.1 * 2.0 + 4.0 * 5.0, 1.1 * 3.0 + 4.0 * 6.0, 1.1 * 4.0 + 4.0 * 7.0)
    assert(Vectors.dense(combination.collectElements) == expectedCombination,
      "DVectorSpace.combine should return the correct result.")
  }

  test("DVectorSpace.dot is implemented properly") {
    val a = sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))), 2)
    val b = sc.parallelize(Array(new DenseVector(Array(5.0, 6.0)), new DenseVector(Array(7.0))), 2)
    val expectedDot = 2.0 * 5.0 + 3.0 * 6.0 + 4.0 * 7.0
    assert(DVectorSpace.dot(a, b) == expectedDot,
      "DVectorSpace.dot should return the correct result.")
  }

  test("DVectorSpace.entrywiseProd is implemented properly") {
    val a = sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))), 2)
    val b = sc.parallelize(Array(new DenseVector(Array(5.0, 6.0)), new DenseVector(Array(7.0))), 2)
    val entrywiseProd = DVectorSpace.entrywiseProd(a, b)
    val expectedEntrywiseProd =
      Vectors.dense(Array(2.0 * 5.0, 3.0 * 6.0, 4.0 * 7.0))
    assert(Vectors.dense(entrywiseProd.collectElements) == expectedEntrywiseProd,
      "DVectorSpace.entrywiseProd should return the correct result.")
  }

  test("DVectorSpace.entrywiseNegDiv is implemented properly") {
    val a = sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))), 2)
    val b = sc.parallelize(Array(new DenseVector(Array(5.0, -6.0)), new DenseVector(Array(0.0))), 2)
    val entrywiseNegDiv = DVectorSpace.entrywiseNegDiv(a, b)
    val expectedEntrywiseNegDiv =
      Vectors.dense(Array(Double.PositiveInfinity, 3.0 / math.abs(-6.0), Double.PositiveInfinity))
    assert(Vectors.dense(entrywiseNegDiv.collectElements) == expectedEntrywiseNegDiv,
      "DVectorSpace.entrywiseNegDiv should return the correct result.")
  }

  test("DVectorSpace.sum is implemented properly") {
    val a = sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))), 2)
    val expectedSum = 2.0 + 3.0 + 4.0
    assert(DVectorSpace.sum(a) == expectedSum,
      "DVectorSpace.sum should return the correct result.")
  }

  test("DVectorSpace.max is implemented properly") {
    val a = sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))), 2)
    val expectedMax = 4.0
    assert(DVectorSpace.max(a) == expectedMax,
      "DVectorSpace.max should return the correct result.")
  }

  test("DVectorSpace.min is implemented properly") {
    val a = sc.parallelize(Array(new DenseVector(Array(2.0, 3.0)), new DenseVector(Array(4.0))), 2)
    val expectedMin = 2.0
    assert(DVectorSpace.min(a) == expectedMin,
      "DVectorSpace.min should return the correct result.")
  }
}
