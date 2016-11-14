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

package org.apache.spark.mllib.optimization.lp.examples

import scala.util.Random

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.optimization.lp.LP
import org.apache.spark.mllib.optimization.lp.VectorSpace._
import org.apache.spark.mllib.optimization.lp.vs.dvector.DVectorSpace
import org.apache.spark.mllib.optimization.lp.vs.vector.DenseVectorSpace
import org.apache.spark.mllib.optimization.lp.fs.dvector.vector.LinopMatrixAdjoint
import org.apache.spark.mllib.random.{RandomDataGenerator, RandomRDDs}
import org.apache.spark.mllib.rdd.RandomVectorRDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.util.random.XORShiftRandom

/**
  * Helper class for generating sparse standard normal values.
  *
  * @param density The density of non sparse values.
  */
private class SparseStandardNormalGenerator(density: Double) extends RandomDataGenerator[Double] {

  private val random = new XORShiftRandom()

  override def nextValue(): Double = if (random.nextDouble < density) random.nextGaussian else 0.0

  override def setSeed(seed: Long): Unit = random.setSeed(seed)

  override def copy(): SparseStandardNormalGenerator = new SparseStandardNormalGenerator(density)
}

/**
  * This example generates a random linear programming problem and solves it using LP.solve
  *
  * The example can be executed as follows:
  * sbt 'test:run-main org.apache.spark.mllib.optimization.lp.examples.TestLPSolver'
  */
object TestLPSolver {
  def main(args: Array[String]) {

    val rnd = new Random(12345)
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("TestLPSolver")
    val sc = new SparkContext(sparkConf)

    val n = 1000 // Transpose constraint matrix row count.
    val m = 100 // Transpose constraint matrix column count.
    val numPartitions = 2

    // Generate the starting vector from uniform distribution U(3.0, 5.0)
    println("generate x")
    val x0 = RandomRDDs.uniformRDD(sc, n, numPartitions).map(v => 3.0 + 2.0 * v).glom.map(new DenseVector(_))

    // Generate the transpose constraint matrix 'B' using sparse uniformly generated values.
    println("generate B")
    val B = new RandomVectorRDD(sc,
      n,
      m,
      numPartitions,
      new SparseStandardNormalGenerator(0.1),
      rnd.nextLong)

    // Generate the cost vector 'c' using uniformly generated values.
    println("generate c")
    val c = RandomRDDs.uniformRDD(sc, n, numPartitions, rnd.nextLong).glom.map(new DenseVector(_))
    // Compute 'b' using the starting 'x' vector.
    println("generate b")
    val b = (new LinopMatrixAdjoint(B))(x0)

    // Solve the linear program using LP.solve, finding the optimal x vector 'optimalX'.
    println("Start solving ...")
    val (optimalVal, _) = LP.solve(c, B, b, sc=sc)
    println("optimalVal: " + optimalVal)
    //println("optimalX: " + optimalX.collectElements.mkString(", "))

    sc.stop()
  }
}
