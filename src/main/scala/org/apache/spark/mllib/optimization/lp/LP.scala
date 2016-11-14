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

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.linalg.CholeskyDecomposition
import org.apache.spark.mllib.optimization.lp.fs.dvector.vector.LinopMatrixAdjoint
import org.apache.spark.mllib.optimization.lp.fs.vector.dvector.LinopMatrix
import org.apache.spark.mllib.optimization.lp.DVectorFunctions._
import org.apache.spark.mllib.optimization.lp.VectorSpace._
import org.apache.spark.mllib.optimization.lp.fs.dvector.dmatrix.SpLinopMatrix

/**
  * An abstract class for solving LP.
  *
  * @param c the objective coefficient DVector.
  * @param rows the constraint DMatrix.
  * @param b the constraint values.
  */
abstract class LP(val c: DVector, val rows: DMatrix, val b: DenseVector) extends Serializable {

  def solve(c: DVector,
            rows: DMatrix,
            b: DenseVector,
            tol: Double,
            maxIter: Int,
            @transient sc: SparkContext): (Double, DVector)
}

object LP extends Logging {

  /**
    * Compute the optimal value and the corresponding vector for LP problem.
    *
    * @param c the objective coefficient DVector.
    * @param rows the constraint DMatrix.
    * @param b the constraint values.
    * @param tol convergence tolerance.
    * @param maxIter maximum number of iterations if it did not converge.
    * @param sc a SparkContext instance.
    * @param row implicit for distributed computations.
    * @param col implicit for local computations.
    * @return optimal value and the corresponding solution vector.
    */
  def solve(c: DVector,
            rows: DMatrix,
            b: DenseVector,
            tol: Double = 1e-8,
            maxIter: Int = 50,
            @transient sc: SparkContext)(
    implicit row: VectorSpace[DVector],
    col: VectorSpace[DenseVector]): (Double, DVector) = {

    row.cache(c)
    val initLocal = Initialize.init(c, rows, b)
    var x: DVector = initLocal._1
    row.cache(x)
    var lambda: DenseVector = initLocal._2
    var lambdaBroadcast = sc.broadcast(lambda)
    var s: DVector = initLocal._3
    row.cache(s)
    val n = initLocal._4
    val m = initLocal._5
    val mu: Double = x.dot(s) / n
    var converged: Boolean = false
    var iter: Int = 1
    var cTx: Double = Double.PositiveInfinity
    val dmat = new LinopMatrix(rows)
    val dmatT = new LinopMatrixAdjoint(rows)
    val etaIter = 0.999
    val cap = 1e20
    val capSqrd = 1e10
    val eps = 1e-20

    while (!converged && iter <= maxIter) {
      println(s"iteration $iter")
      // B^T * x - b
      var rb: DenseVector = col.combine(1.0, dmatT(x), -1.0,  b)
      var rc: DVector = row.combine(1.0, dmat(lambdaBroadcast.value), 1.0, s.diff(c))
      row.cache(rc)
      //rc.localCheckpoint()
      // D = X^(1/2) * S^(-1/2)
      val D: DVector = row.entrywiseProd(
        x.mapElements {
          case a if math.abs(a) < eps => math.signum(a) * capSqrd
          case a if a >= 0.0 => math.sqrt(a)
        },
        s.mapElements {
          case a if 0 < a && a < eps => capSqrd
          case a if a >= eps => 1 / math.sqrt(a)
        }
      )

      val D2: DVector = row.entrywiseProd(
        x,
        s.mapElements {
          case a if math.abs(a) < eps => math.signum(a) * cap
          case a if math.abs(a) >= eps => math.pow(a, -1)
        }
      )
      row.cache(D2)
      //D2.localCheckpoint()
      // solve (14.30) for (dxAff, dLambdaAff, dsAff)
      // 1) solve for BTD2B dLambdaAff = -rb + BT * (-D^2 * rc + x)
      val DB: DMatrix = (new SpLinopMatrix(D))(rows)
      val DBRowMat: LPRowMatrix = new LPRowMatrix(DB, n, m)
      val BTD2B: BDV[Double] = DBRowMat.computeGramianMatrixColumn(m, depth=2)
      val BTD2rcx = dmatT(x.diff(row.entrywiseProd(D2, rc)))
      val dLambdaAffRightSide: Vector = col.combine(1.0, BTD2rcx, -1.0, rb)
      val upTriArray: Array[Double] = BTD2B.data
      // capturing side effects:
      val upTriArrayCopy = upTriArray.clone()
      val dLambdaAffArray = dLambdaAffRightSide.toArray
      CholeskyDecomposition.solve(upTriArrayCopy, dLambdaAffArray) // inplace dLambdaAffArray
      val dLambdaAff: DenseVector = new DenseVector(dLambdaAffArray)
      // 2) dsAff = -rc - B * dLambdaAff
      val dsAff: DVector = row.combine(-1.0, rc, -1.0, dmat(dLambdaAff))
      // 3) dxAff = -x - D^2 * dsAff
      val dxAff: DVector = row.combine(-1.0, x, -1.0, row.entrywiseProd(D2, dsAff))
      // Calculate following Doubles alphaPriAff, alphaDualAff, muAff (14.32), (14.33)
      val alphaPriAff: Double = math.min(1.0, row.min(row.entrywiseNegDiv(x, dxAff)))
      val alphaDualAff: Double = math.min(1.0, row.min(row.entrywiseNegDiv(s, dsAff)))
      val muAff: Double = {
        val nx: DVector = row.combine(1.0, x, alphaPriAff, dxAff)
        val ns: DVector = row.combine(1.0, s, alphaDualAff, dsAff)
        row.dot(nx, ns) / n
      }
      val sigma: Double = math.pow(muAff / mu, 3) // heuristic
      println("sigma = " + sigma)
      // Solve (14.35) for (dx, dLambda, ds)
      // 1) BTD2B dLambda = -rb + BT * D2 *(-rc + s + X^(-1) dXAff dSAff e - sigma mu X^(-1)e)
      val xinv: DVector = x.mapElements {
          case a if 0 < math.abs(a) && math.abs(a) < eps => math.signum(a) * cap
          case a if a >= eps => math.pow(a, -1)
          case _ => throw new IllegalArgumentException("Found zero element in X")
        }
      val xinvdXAffdsAff: DVector = row.entrywiseProd(xinv, row.entrywiseProd(dxAff, dsAff))
      val dLambdaRightSide: DenseVector = col.combine(
        -1.0,
        rb,
        1.0,
        dmatT(
          row.entrywiseProd(D2,
            row.combine(
              1.0,
              row.combine(1.0, s.diff(rc), 1.0, xinvdXAffdsAff),
              -1.0 * sigma * mu,
              xinv
            )
          )
        )
      )

      val dLambdaArray: Array[Double] = CholeskyDecomposition.solve(upTriArray, dLambdaRightSide.toArray)
      val dLambda: DenseVector = new DenseVector(dLambdaArray)
      // 2) ds = -rc - B * dLambda
      val ds: DVector = row.combine(-1.0, rc, -1.0, dmat(dLambda))
      // 3) dx = -D^2 dS e - x - S^(-1) dXAff dSAff e + sigma mu S^(-1) e
      val sinv: DVector = s.mapElements {
        case a if math.abs(a) < eps => math.signum(a) * cap
        case a if math.abs(a) >= eps => math.pow(a, -1)
        //case _ => throw new IllegalArgumentException("Zero element in s")
      }
      val sinvdXAffdsAff: DVector = row.entrywiseProd(sinv, row.entrywiseProd(dxAff, dsAff))
      val dx: DVector = row.combine(
        1.0,
        row.combine(
          1.0,
          row.combine(-1.0, row.entrywiseProd(D2, ds), -1.0, x),
          -1.0,
          sinvdXAffdsAff
        ),
        sigma * mu,
        sinv
      )
      val alphaPriIterMax: Double = row.min(row.entrywiseNegDiv(x, dx))
      val alphaDualIterMax: Double = row.min(row.entrywiseNegDiv(s, ds))
      val alphaPriIter: Double = math.min(1.0, etaIter * alphaPriIterMax)
      val alphaDualIter: Double = math.min(1.0, etaIter * alphaDualIterMax)
      // x = x + alphaPriIter * dx
      x = row.combine(1.0, x, alphaPriIter, dx)
      //x.checkpoint()
      x.localCheckpoint()
      // lambda = lambda + alphaDualIter * dLambda
      lambda = new DenseVector((lambdaBroadcast.value.toBreeze + alphaDualIter * dLambda.toBreeze).toArray)
      lambdaBroadcast = sc.broadcast(lambda)
      // s = s + alphaDualIter * ds
      s = row.combine(1.0, s, alphaDualIter, ds)
      //s.checkpoint()
      s.localCheckpoint()
      // page 226: LP Wright --> differ in condition 3 (Check with Mehrotra's original paper and others)
      rb = col.combine(1.0, dmatT(x), -1.0,  b)
      rc = row.combine(1.0, dmat(lambdaBroadcast.value), 1.0, s.diff(c))
      cTx = c.dot(x)
      val bTlambda = col.dot(b, lambdaBroadcast.value)
      val covg1 = math.sqrt(col.dot(rb, rb)) / (1 + math.sqrt(col.dot(b, b)))
      val covg2 = math.sqrt(rc.dot(rc)) / (1 + math.sqrt(c.dot(c)))
      val covg3 = math.abs(cTx - bTlambda) / (1 + math.abs(bTlambda))
      converged = (covg1 < tol) && (covg2 < tol) && (covg3 < tol)
      println("first conv condition: " + covg1)
      println("second conv condition: " + covg2)
      println("third conv condition: " + covg3)
      println("cTx: " + cTx)
      println("b dot lambda: " + bTlambda)
      iter += 1
    }

    (cTx, x)
  }
}