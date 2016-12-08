# spark-lp (WIP)

This package offers an implementation of [Mehrohra's predictor-corrector interior point algorithm](https://en.wikipedia.org/wiki/Mehrotra_predictor%E2%80%93corrector_method), described in [numerical optimization]([http://www.springer.com/gp/book/9780387303031]), on top of Apache Spark to solve large-scale [linear programming](https://en.wikipedia.org/wiki/Linear_programming) problems.

Linear programming has the following standard form: 

	minimize c^T x 
	subject to Ax=b and x >= 0

where `c, b` are given vectors ((.)^T is the traspose operation), `A` is a given `m` by `n` matrix and `x` is the objective vector. We assume that `A` the number of rows (equations) is
at most equal to the number of columns (unknowns) (`m <= n`) and `A` has full row rank, thus `AA^T` is invertible.

## Example

The following is an example of using spark-lp to solve a linear programming problem

	import org.apache.spark.{SparkConf, SparkContext}
	import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
	import org.apache.spark.mllib.optimization.lp.VectorSpace._
	import org.apache.spark.mllib.optimization.lp.vs.dvector.DVectorSpace
	import org.apache.spark.mllib.optimization.lp.vs.vector.DenseVectorSpace
	import org.apache.spark.mllib.optimization.lp.LP

	val sparkConf = new SparkConf().setMaster("local[2]").setAppName("TestLPSolver")
	val sc = new SparkContext(sparkConf)

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

	val c: DVector = sc.parallelize(cArray, numPartitions).glom.map(new DenseVector(_))
	val rows: DMatrix = sc.parallelize(BArray, numPartitions).map(Vectors.dense(_))
	val b: DenseVector = new DenseVector(bArray)

	val (v, x): (Double, DVector) = LP.solve(c, rows, b, sc=sc)
	val xx = Vectors.dense(x.flatMap(_.toArray).collect())
	println(s"optimial vector is $xx")
	println("optimal min value: " + v)

## Software Architecture Overview

Our implementation was inspired by [spark-tfocs](https://github.com/databricks/spark-tfocs). The spark-tfocs architecture design is the best suitable option for separating local and distributed vector space operations by means of

* `DenseVector` A wrapper around `Array[Double]` with vector operations support. (Imported
  from `org.apache.spark.mllib.linalg`)

* `DVector` A distributed vector, stored as an `RDD[DenseVector]`, where each partition comprises a single `DenseVector` containing a slice of the complete distributed vector. This has the advantage of using BLAS operations directly as opposed to `RDD[Double]`. More information is available in `org.apache.spark.mllib.optimization.lp.VectorSpace`.

* `DMatrix` A distributed matrix, stored as an `RDD[Vector]`, where each (possibly sparse) `Vector`
  represents a row of the matrix. More information is available in
  `org.apache.spark.mllib.optimization.lp.VectorSpace`.

## Advantages

* spark-lp is able to solve large-scale LP problems in a distributed way with fault-tolerance over commodity clusters of machines. (Stay tuned for more results!)

* spark-lp is ~100X faster and more accurate than spark-tfocs for solving large-scale LP problems. (Stay tuned for the published results)

## TODOs:

* Add preprocessing to capture more general LP formats.
* Add infeasibility detection.
* Extend to QP solver.
