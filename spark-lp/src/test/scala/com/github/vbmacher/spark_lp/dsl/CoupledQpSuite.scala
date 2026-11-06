package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.dsl.implicits._
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.scalatest.funsuite.AnyFunSuite
import java.nio.file.Files

class CoupledQpSuite extends AnyFunSuite with DataFrameSuiteBase {

  test("off-diagonal PSD factors have an independent original-space KKT witness") {
    implicit val ss: SparkSession = spark
    Seq(NewtonSolver.Cholesky, NewtonSolver.ConjugateGradient).foreach { backend =>
      val model = LpProblem("coupled")
      val x = model.variable("x")
      val y = model.variable("y")
      model += QpObjective.squared(x + 2.0 * y - 5.0) + QpObjective.squared(x - y - 1.0)

      val result = model.solve(SolveConfig(newtonSolver = backend, maxIterations = 100))
      try {
        assert(result.status == LpStatus.Optimal)
        val xValue = result.value(x)
        val yValue = result.value(y)
        // F=[[1,2],[1,-1]], target=[5,1], Q=2F^T F has nonzero off-diagonal.
        val firstDeviation = xValue + 2.0 * yValue - 5.0
        val secondDeviation = xValue - yValue - 1.0

        assert(math.abs(xValue - 7.0/3) < 1e-4)
        assert(math.abs(yValue - 4.0/3) < 1e-4)
        assert(math.abs(2.0 * firstDeviation + 2.0 * secondDeviation) < 1e-4)
        assert(math.abs(4.0 * firstDeviation - 2.0 * secondDeviation) < 1e-4)
        assert(math.abs(result.objectiveValue - (firstDeviation * firstDeviation + secondDeviation * secondDeviation)) < 1e-7)
      } finally result.close()
    }
  }

  test("coupled-QP solutions can be reused as starts") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("coupled start")
    val x = model.variable("x")
    model += QpObjective.squared(x - 2.0)

    val first = model.solve()
    val start = first.asStart()
    try {
      val second = model.solve(SolveConfig(start = Some(start)))
      try {
        assert(second.start.exists(_.disposition == LpStartDisposition.Accepted))
        assert(math.abs(second.value(x) - 2.0) < 1e-4)
      } finally second.close()
    } finally { start.close(); first.close() }
  }

  test("coupled-QP JSON solution candidates contain only declared variables") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("coupled JSON")
    val x = model.variable("x")
    model += QpObjective.squared(x - 2.0)

    val result = model.solve()
    val directory = Files.createTempDirectory("spark-lp-coupled-json-").resolve("model").toString
    try {
      LpJson.write(model, directory, Some(LpSolutionData.fromSolution(result)))
      val document = LpJson.read(directory)
      val imported = document.model.toProblem()
      val values = document.solution.get.values.get

      assert(values.count() == 1)

      val report = imported.model.validateCandidate(values)
      try {
        assert(report.feasible)
      } finally report.close()
    } finally {
      result.close()
      val path = new org.apache.hadoop.fs.Path(directory)
      path.getFileSystem(spark.sparkContext.hadoopConfiguration).delete(path.getParent, true)
    }
  }

  test("rank-deficient PSD, free variables, shifts, fixed variables and objective sense") {
    implicit val ss: SparkSession = spark
    for (backend <- Seq(NewtonSolver.Auto, NewtonSolver.ConjugateGradient); sense <- Seq(Minimize, Maximize)) {
      val model = LpProblem("transformed coupled", sense)
      val x = model.variable("free", lowerBound = Double.NegativeInfinity)
      val y = model.variable("shifted", lowerBound = -2, upperBound = Some(4.0))
      val fixed = model.variable("fixed", lowerBound = 2, upperBound = Some(2.0))
      val objective = QpObjective.squared(x + y + fixed - 5.0)
      model += objective * (if (sense == Minimize) 1.0 else -1.0)
      model += (x - y === 1.0)

      val result = model.solve(SolveConfig(newtonSolver = backend, maxIterations = 100))
      try {
        assert(result.status == LpStatus.Optimal)
        assert(math.abs(result.value(x) - 2) < 2e-4)
        assert(math.abs(result.value(y) - 1) < 2e-4)
        assert(result.value(fixed) == 2.0)
        assert(math.abs(result.objectiveValue) < 1e-7)
      } finally result.close()
    }
  }

  test("keyed sparse factors aggregate repeated terms and repeated solves do not mutate the model") {
    implicit val ss: SparkSession = spark
    val localSpark = spark
    import localSpark.implicits._
    val model = LpProblem("keyed factors")
    val vars = model.variables("x", Seq(("a", 1.0), ("b", 2.0)).toDF("key", "a"), col("key"))
    model += QpObjective.squared(vars.sum(col("a")) + vars.sum(col("a")) - 6.0)
    model += (vars.sum === 2.0)
    val handlesBefore = model.handles.size
    val constraintsBefore = model.constraints.size
    for (_ <- 1 to 2) {
      val result = model.solve()
      try {
        assert(result.status == LpStatus.Optimal)
        val values = result.values(vars).select("lp_value").collect().map(_.getDouble(0))

        assert(values.forall(v => math.abs(v - 1.0) < 1e-4))
        assert(model.handles.size == handlesBefore)
        assert(model.constraints.size == constraintsBefore)
      } finally result.close()
    }
    model.setObjective(vars.sum)

    val linear = model.solve()
    try {
      assert(math.abs(linear.objectiveValue - 2) < 1e-6)
    } finally linear.close()
  }

  test("invalid factor sense and integer categories are rejected; zero factors use the LP path") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("invalid factors")
    val x = model.variable("x")
    model += QpObjective.squared(x) * -1.0
    model += (x >= 1.0)

    intercept[LpModelException](model.solve())
    model.setObjective(QpObjective.squared(x, 0.0) + x)

    val linear = model.solve()
    try {
      assert(math.abs(linear.objectiveValue - 1) < 1e-6)
    } finally linear.close()
    val integer = LpProblem("integer")
    val i = integer.variable("i", upperBound = Some(2.0), category = Integer)
    integer += QpObjective.squared(i)

    intercept[LpModelException](integer.solve())
  }
  test("rank-deficient bounded factors support Cholesky, while free factors select regularized CG") {
    implicit val ss: SparkSession = spark
    val model = LpProblem("rank one")
    val x = model.variable("x")
    val y = model.variable("y")
    model += QpObjective.squared(x + y - 3.0)
    model += (x - y === 1.0)

    val result = model.solve(SolveConfig(newtonSolver = NewtonSolver.Cholesky))
    try {
      assert(result.status == LpStatus.Optimal)
      assert(math.abs(result.value(x) - 2) < 1e-4)
      assert(math.abs(result.value(y) - 1) < 1e-4)
    } finally result.close()
    val freeModel = LpProblem("free")
    val f = freeModel.variable("f", lowerBound = Double.NegativeInfinity)
    freeModel += QpObjective.squared(f - 2.0)

    val error = intercept[LpModelException](freeModel.solve(SolveConfig(newtonSolver = NewtonSolver.Cholesky)))

    assert(error.getMessage.contains("requires regularized ConjugateGradient"))
  }

}
