package com.github.vbmacher.spark_lp.dsl

import com.github.vbmacher.spark_lp.Numerics
import org.apache.spark.rdd.RDD

/**
  * Original constraint row before solver transformations.
  *
  * @param name display name of the expanded row.
  * @param group encoded group key for a generated row, or `None` for a scalar row.
  * @param sense comparison symbol: `<=`, `>=`, or `=`.
  * @param rhs right-hand side in original model units.
  */
final case class EvidenceRow(
  name: String,
  group: Option[String],
  sense: String,
  rhs: Double
)

/**
  * Original variable data used to verify a certificate independently of solver transformations.
  *
  * @param name display name of the variable.
  * @param lower inclusive lower bound.
  * @param upper inclusive upper bound, or `None` when unbounded above.
  * @param cost linear objective coefficient in original objective direction.
  * @param coefficients original constraint-row index to coefficient.
  * @param curvature diagonal quadratic coefficient in `0.5 * curvature * x^2` form.
  */
final case class EvidenceVariable(
  name: String,
  lower: Double,
  upper: Option[Double],
  cost: Double,
  coefficients: Map[Int, Double],
  curvature: Double = 0.0
)

/**
  * Materialized original-model snapshot used for independent evidence verification.
  *
  * @param rows constraint rows in the index order referenced by variable coefficients.
  * @param variables distributed variable identity and evidence data.
  * @param sense original objective direction.
  */
final case class EvidenceModel(
  rows: IndexedSeq[EvidenceRow],
  variables: RDD[((Int, String), EvidenceVariable)],
  sense: ObjectiveSense
)

/**
  * Result of checking certificate equations against an evidence model.
  *
  * @param valid true when absolute and scale-aware equation, sign and separation checks pass.
  * @param residual largest absolute violation encountered while checking the certificate.
  * @param margin normalized certificate margin; one is required for infeasibility and a positive
  *               improvement is required for an unbounded direction.
  */
final case class EvidenceVerification(
  valid: Boolean,
  residual: Double,
  margin: Double
)

sealed trait LpEvidence extends AutoCloseable {
  def model: EvidenceModel

  def verify(tolerance: Double = 1e-8): EvidenceVerification
}

/** Farkas convention: A^T y + lowerMultiplier + upperMultiplier = 0,
  * b^T y + l^T lowerMultiplier + u^T upperMultiplier = 1.
  * <= row and upper-bound multipliers are nonpositive; >= and lower are nonnegative.
  * Equality multipliers are unrestricted. All arrays are in original units.
  *
  * @param model original model against which the certificate is checked.
  * @param rows row multipliers in the same order as `model.rows`.
  * @param bounds lower- and upper-bound multipliers keyed by variable identity.
  */
final case class InfeasibilityCertificate(
  model: EvidenceModel,
  rows: IndexedSeq[Double],
  bounds: RDD[((Int, String), (Double, Double))]
) extends LpEvidence {
  override def verify(tolerance: Double): EvidenceVerification =
    LpEvidenceVerifier.infeasibility(this, tolerance)

  override def close(): Unit = {
    model.variables.unpersist(false)
    bounds.unpersist(false)
  }
}

/** Direction in original units, normalized to an improvement of one. A feasible point is
  * present only for proven unboundedness; a direction alone does not prove primal feasibility.
  *
  * @param model original model against which the direction is checked.
  * @param direction variable displacement that preserves feasibility and improves the objective.
  * @param point feasible base point proving that the direction yields an unbounded feasible ray.
  */
final case class UnboundedDirection(
  model: EvidenceModel,
  direction: RDD[((Int, String), Double)],
  point: Option[RDD[((Int, String), Double)]]
) extends LpEvidence {
  override def verify(tolerance: Double): EvidenceVerification =
    LpEvidenceVerifier.unbounded(this, tolerance)

  override def close(): Unit = {
    model.variables.unpersist(false)
    direction.unpersist(false)
    point.foreach(_.unpersist(false))
  }
}

/** Checks original coefficients, bounds and objective, without trusting solver residuals/status.
  * `valid` checks both absolute and homogeneous residuals at the requested tolerance;
  * it is a numerical witness, not an exact-arithmetic proof.
  */
object LpEvidenceVerifier {
  private def checked(v: Double): Double = if (Numerics.isFinite(v)) math.max(0.0, v) else Double.PositiveInfinity

  private def boundScale(bound: Double): Double = math.max(1.0, math.abs(bound))

  private def columnScale(variable: EvidenceVariable): Double = {
    val scale = math.max(math.abs(variable.cost), math.max(variable.coefficients.values.map(math.abs).sum,
      if (!variable.lower.isNegInfinity || variable.upper.nonEmpty) 1.0 else 0.0))
    if (scale == 0.0) 1.0 else scale
  }

  private def checkTolerance(t: Double): Unit = require(Numerics.isFinite(t) && t > 0, "tolerance must be finite and positive")

  private def keysMatch[A: scala.reflect.ClassTag, B: scala.reflect.ClassTag](a: RDD[((Int, String), A)], b: RDD[((Int, String), B)]): Boolean =
    a.mapValues(_ => 1).cogroup(b.mapValues(_ => 1)).filter { case (_, (x, y)) => x.size != 1 || y.size != 1 }.take(1).isEmpty

  def infeasibility(proof: InfeasibilityCertificate, tolerance: Double = 1e-8): EvidenceVerification = {
    checkTolerance(tolerance)
    val model = proof.model
    val y = proof.rows
    if (y.size != model.rows.size || !y.forall(Numerics.isFinite) || !keysMatch(model.variables, proof.bounds))
      return EvidenceVerification(false, Double.PositiveInfinity, Double.NaN)
    val rowNorms = model.variables.values.flatMap(_.coefficients.map { case (r, a) => r -> math.abs(a) })
      .reduceByKey(_ + _).collectAsMap()
    val rowScales = model.rows.zipWithIndex.map { case (r, i) =>
      val scale = math.max(math.abs(r.rhs), rowNorms.getOrElse(i, 0.0))
      if (scale == 0.0) 1.0 else scale
    }
    val rowError = model.rows.zip(y).map { case (r, v) =>
      r.sense match {
        case "<=" => checked(v);
        case ">=" => checked(-v);
        case _ => 0.0
      }
    }.foldLeft(0.0)(math.max)
    val joined = model.variables.join(proof.bounds)
    val multiplierNorm = math.max(y.zip(rowScales).map { case (v, scale) => math.abs(v) * scale }.foldLeft(0.0)(math.max),
      joined.values.map { case (v, (lower, upper)) =>
        math.max(math.abs(lower) * (if (v.lower.isNegInfinity) 1.0 else boundScale(v.lower)),
          math.abs(upper) * v.upper.map(boundScale).getOrElse(1.0))
      }
        .fold(0.0)(math.max))
    val terms = joined.map { case (_, (v, (lower, upper))) =>
      val ay = v.coefficients.iterator.map { case (r, a) => a * y(r) }.sum
      val sign = math.max(checked(-lower), checked(upper))
      val absent = math.max(if (v.lower.isNegInfinity) math.abs(lower) else 0.0,
        if (v.upper.isEmpty) math.abs(upper) else 0.0)
      val stationarity = math.abs(ay + lower + upper)
      val err = Seq(sign, absent, stationarity).map(checked).max
      val loScale = if (v.lower.isNegInfinity) 1.0 else boundScale(v.lower)
      val hiScale = v.upper.map(boundScale).getOrElse(1.0)
      val stationarityScale = v.coefficients.map { case (r, a) => math.abs(a) / rowScales(r) }.sum *
        multiplierNorm + math.abs(lower) + math.abs(upper)
      val relativeSigns = Seq(checked(-lower) * loScale, checked(upper) * hiScale, absent).max
      val relativeError = math.max(Numerics.relativeResidual(relativeSigns, multiplierNorm),
        Numerics.relativeResidual(stationarity, stationarityScale))
      val margin = (if (v.lower.isNegInfinity) 0.0 else lower * v.lower) +
        v.upper.map(_ * upper).getOrElse(0.0)
      val boundMarginScale = (if (v.lower.isNegInfinity) 0.0 else math.abs(v.lower) / loScale) +
        v.upper.map(u => math.abs(u) / hiScale).getOrElse(0.0)
      (err, margin, relativeError, boundMarginScale)
    }.fold((0.0, 0.0, 0.0, 0.0)) { case ((e, m, r, s), (f, n, t, u)) =>
      (math.max(e, f), m + n, math.max(r, t), s + u)
    }
    val margin = terms._2 + model.rows.zip(y).map { case (r, v) => r.rhs * v }.sum
    val residual = math.max(math.max(rowError, terms._1), checked(math.abs(margin - 1.0)))
    val relativeRowError = model.rows.zipWithIndex.map { case (r, i) =>
      val error = r.sense match {
        case "<=" => checked(y(i))
        case ">=" => checked(-y(i))
        case _ => 0.0
      }
      Numerics.relativeResidual(error * rowScales(i), multiplierNorm)
    }.foldLeft(0.0)(math.max)
    val relativeError = math.max(terms._3, relativeRowError)
    val rhsScale = terms._4 + model.rows.zip(rowScales).map { case (r, scale) => math.abs(r.rhs) / scale }.sum
    val separation = Numerics.relativeResidual(margin, rhsScale * multiplierNorm)
    EvidenceVerification(residual <= tolerance && relativeError <= tolerance &&
      Numerics.isFinite(separation) && separation > tolerance && margin > tolerance, residual, margin)
  }

  def unbounded(proof: UnboundedDirection, tolerance: Double = 1e-8): EvidenceVerification = {
    checkTolerance(tolerance)
    val model = proof.model

    def violations(values: RDD[((Int, String), Double)], ray: Boolean): (Double, Double, Double, Double) = {
      if (!keysMatch(model.variables, values))
        return (Double.PositiveInfinity, Double.NaN, Double.PositiveInfinity, Double.PositiveInfinity)
      val joined = model.variables.join(values)
      val directionNorm = if (ray) joined.values.map { case (v, x) => math.abs(x) * columnScale(v) }
        .fold(0.0)(math.max) else 1.0
      val scalar = joined.map { case (_, (v, x)) =>
        val scale = if (ray) columnScale(v) else 1.0
        val lo = if (v.lower.isNegInfinity) 0.0 else (if (ray) -x else v.lower - x)
        val hi = v.upper.map(u => if (ray) x else x - u).getOrElse(0.0)
        val curvatureError = if (ray) checked(math.abs(v.curvature * x)) else 0.0
        val relativeError = if (ray) math.max(
          Numerics.relativeResidual(math.max(checked(lo), checked(hi)) * scale, directionNorm),
          if (v.curvature > 0.0) Numerics.relativeResidual(math.abs(x) * scale, directionNorm) else 0.0) else 0.0
        (if (Numerics.isFinite(x)) math.max(curvatureError, math.max(checked(lo), checked(hi)))
        else Double.PositiveInfinity, v.cost * x, relativeError, math.abs(v.cost) / scale)
      }.fold((0.0, 0.0, 0.0, 0.0)) { case ((a, b, e, f), (c, d, g, h)) =>
        (math.max(a, c), b + d, math.max(e, g), f + h)
      }
      val activities = joined.flatMap { case (_, (v, x)) =>
        val scale = if (ray) columnScale(v) else 1.0
        v.coefficients.map { case (r, a) => (r, (a * x, math.abs(a) / scale)) }
      }.reduceByKey { case ((a, b), (c, d)) => (a + c, b + d) }.collectAsMap()
      val rowErrors = model.rows.zipWithIndex.map { case (r, i) =>
        val (activity, scale) = activities.getOrElse(i, (0.0, 0.0))
        val d = activity - (if (ray) 0.0 else r.rhs)
        val error = checked(LpSense.violation(r.sense, d))
        (error, if (ray) Numerics.relativeResidual(error, scale * directionNorm) else 0.0)
      }.foldLeft((0.0, 0.0)) { case ((a, b), (c, d)) => (math.max(a, c), math.max(b, d)) }
      (math.max(scalar._1, rowErrors._1), scalar._2, math.max(scalar._3, rowErrors._2),
        scalar._4 * directionNorm)
    }

    val ray = violations(proof.direction, ray = true)
    val margin = (if (model.sense == Minimize) -1.0 else 1.0) * ray._2
    val pointError = proof.point.map(violations(_, ray = false)._1).getOrElse(Double.PositiveInfinity)
    val residual = math.max(math.max(ray._1, pointError), checked(math.abs(margin - 1.0)))
    val separation = Numerics.relativeResidual(margin, ray._4)
    EvidenceVerification(residual <= tolerance && ray._3 <= tolerance &&
      Numerics.isFinite(separation) && separation > tolerance && margin > tolerance, residual, margin)
  }
}
