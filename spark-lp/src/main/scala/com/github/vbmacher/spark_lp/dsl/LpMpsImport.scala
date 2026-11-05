package com.github.vbmacher.spark_lp.dsl

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.collection.mutable
import org.apache.spark.sql.SparkSession
import com.github.vbmacher.spark_lp.dsl.implicits._

/**
  * Selects MPS data sets and bounds driver-side parsing work.
  *
  * @param rhsSet RHS set to read, or the first set encountered when absent.
  * @param boundSet bounds set to read, or the first set encountered when absent.
  * @param rangeSet range set to read, or the first set encountered when absent.
  * @param objectiveSense explicit objective direction overriding the file declaration.
  * @param maxBytes maximum accepted input-file size.
  * @param maxVariables maximum distinct column names.
  * @param maxRows maximum non-objective row declarations.
  * @param maxCoefficients maximum distinct matrix coefficient positions.
  */
final case class MpsReadOptions(
  rhsSet: Option[String] = None,
  boundSet: Option[String] = None,
  rangeSet: Option[String] = None,
  objectiveSense: Option[ObjectiveSense] = None,
  maxBytes: Long = 64L * 1024 * 1024,
  maxVariables: Int = 100000,
  maxRows: Int = 10000,
  maxCoefficients: Int = 1000000
) {
  require(maxBytes > 0 && maxVariables > 0 && maxRows > 0 && maxCoefficients > 0, "MPS parsing limits must be positive")
}

/**
  * Imported MPS model and mappings back to source names.
  *
  * @param model reconstructed spark-lp problem.
  * @param variables source column name to reconstructed scalar variable.
  * @param constraints source row name to one or more reconstructed constraints; ranged rows produce
  *                    two constraints.
  * @param rhsSet RHS set selected during import.
  * @param boundSet bounds set selected during import.
  * @param rangeSet range set selected during import.
  */
final case class LpMpsModel(
  model: LpProblem,
  variables: Map[String, LpVariable],
  constraints: Map[String, Vector[LpConstraint]],
  rhsSet: Option[String],
  boundSet: Option[String],
  rangeSet: Option[String]
)

/** Bounded driver-side free/fixed-whitespace MPS parser. Does not invoke a solver. */
object LpMpsImport {
  /**
    * Mutable bounds accumulated for one MPS column while parsing.
    *
    * @param lower current inclusive lower bound.
    * @param upper current inclusive upper bound, or `None` when unbounded above.
    * @param category current variable category.
    * @param markerInteger true when the column was declared inside an `INTORG` marker region.
    * @param explicitLower true after a bound record explicitly sets the lower side.
    * @param explicitUpper true after a bound record explicitly sets the upper side.
    */
  private final case class Bounds(
    var lower: Double = 0.0,
    var upper: Option[Double] = None,
    var category: VariableCategory = Continuous,
    markerInteger: Boolean = false,
    var explicitLower: Boolean = false,
    var explicitUpper: Boolean = false
  )

  def read(path: Path, options: MpsReadOptions = MpsReadOptions())(implicit spark: SparkSession): LpMpsModel = {
    if (Files.size(path) > options.maxBytes) throw new LpModelException(s"MPS exceeds maxBytes=${options.maxBytes}: $path")
    val reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)
    val rows = mutable.LinkedHashMap.empty[String, String]
    val columns = mutable.LinkedHashMap.empty[String, Bounds]
    val coefficients = mutable.LinkedHashMap.empty[(String, String), Double]
    val rhs = mutable.Map.empty[String, Double]
    val ranges = mutable.Map.empty[String, Double]
    val rhsSets = mutable.Set.empty[String]
    val boundSets = mutable.Set.empty[String]
    val rangeSets = mutable.Set.empty[String]
    var chosenRhs = options.rhsSet
    var chosenBounds = options.boundSet
    var chosenRanges = options.rangeSet
    var section = ""
    var modelName = path.getFileName.toString
    var sense: ObjectiveSense = Minimize
    var objective = ""
    var objectiveName: Option[String] = None
    var integerRegion = false
    var ended = false
    var lineNumber = 0
    var records = 0
    val seenSections = mutable.Set.empty[String]
    val rank = Map("NAME" -> 0, "OBJSENSE" -> 1, "OBJNAME" -> 1, "ROWS" -> 2, "COLUMNS" -> 3,
      "RHS" -> 4, "RANGES" -> 5, "BOUNDS" -> 6, "ENDATA" -> 7)

    def fail(message: String): Nothing = throw new LpModelException(s"MPS $path:$lineNumber [$section]: $message")

    def number(text: String): Double = {
      val n = try text.replace('D', 'E').replace('d', 'e').toDouble catch {
        case _: NumberFormatException => fail(s"Invalid number '$text'")
      }
      if (!java.lang.Double.isFinite(n)) fail(s"Non-finite numeric entry '$text'; use FR/MI/PL bounds")
      n
    }

    def readSense(token: String): Unit = token.toUpperCase(java.util.Locale.ROOT) match {
      case "MIN" | "MINIMIZE" => sense = Minimize
      case "MAX" | "MAXIMIZE" => sense = Maximize
      case _ => fail(s"Unsupported objective sense '$token'")
    }

    def rowExists(name: String): Unit = if (!rows.contains(name)) fail(s"Unknown row '$name'")

    def pairs(tokens: Vector[String]): Vector[(String, Double)] = {
      if (tokens.length != 3 && tokens.length != 5) fail("Expected name followed by one or two row/value pairs")
      tokens.tail.grouped(2).map { p => rowExists(p(0)); p(0) -> number(p(1)) }.toVector
    }

    try {
      var raw = reader.readLine()
      while (raw != null) {
        lineNumber += 1
        val line = raw.trim
        if (line.nonEmpty && !line.startsWith("*")) {
          if (ended) fail("Unexpected content after ENDATA")
          val tokens = line.split("\\s+").toVector
          val first = tokens.head
          val isHeader = rank.contains(first) && (tokens.size == 1 ||
            Set("NAME", "OBJSENSE", "OBJNAME")(first) && !raw.head.isWhitespace)
          if (isHeader) {
            if (seenSections(first)) fail(s"Repeated section $first")
            if (section.nonEmpty && rank(first) < rank(section)) fail(s"Out-of-order section $first")
            if (section == "COLUMNS" && integerRegion) fail("INTORG marker has no matching INTEND")
            section = first;
            seenSections += first
            first match {
              case "NAME" => if (tokens.size > 2) fail("NAME accepts one model name") else modelName = tokens.lift(1).getOrElse(modelName)
              case "OBJSENSE" => if (tokens.size == 2) readSense(tokens(1)) else if (tokens.size > 2) fail("Invalid OBJSENSE")
              case "OBJNAME" => if (tokens.size == 2) objectiveName = Some(tokens(1)) else if (tokens.size > 2) fail("Invalid OBJNAME")
              case "ENDATA" => ended = true
              case _ => ()
            }
          } else section match {
            case "OBJSENSE" => if (tokens.size != 1) fail("Expected MIN or MAX") else readSense(first)
            case "OBJNAME" => if (tokens.size != 1) fail("Expected objective row name") else objectiveName = Some(first)
            case "ROWS" =>
              if (tokens.size != 2 || !Set("N", "E", "L", "G")(first)) fail("Expected row type N/E/L/G and name; unsupported sections (including SOS) are rejected")
              if (rows.contains(tokens(1))) fail(s"Duplicate row '${tokens(1)}'")
              if (rows.size >= options.maxRows + 1) fail(s"Exceeded maxRows=${options.maxRows}")
              if (first == "N") {
                if (objective.nonEmpty) fail("Only one free objective row is supported")
                objective = tokens(1)
              }
              rows(tokens(1)) = first
            case "COLUMNS" =>
              if (tokens.lift(1).contains("'MARKER'")) {
                if (tokens.size != 3) fail("Invalid integer marker")
                tokens(2) match {
                  case "'INTORG'" if !integerRegion => integerRegion = true
                  case "'INTEND'" if integerRegion => integerRegion = false
                  case _ => fail("Unbalanced or unsupported integer marker")
                }
              } else {
                val entries = pairs(tokens)
                if (!columns.contains(first) && columns.size >= options.maxVariables) fail(s"Exceeded maxVariables=${options.maxVariables}")
                val b = columns.getOrElseUpdate(first,
                  Bounds(category = if (integerRegion) Integer else Continuous, markerInteger = integerRegion))
                if ((b.category == Integer) != integerRegion) fail(s"Column '$first' appears inside and outside integer markers")
                entries.foreach { case (row, value) =>
                  records += 1
                  if (records > options.maxCoefficients) fail(s"Exceeded maxCoefficients=${options.maxCoefficients}")
                  val key = first -> row
                  val sum = coefficients.getOrElse(key, 0.0) + value
                  if (!java.lang.Double.isFinite(sum)) fail(s"Coefficient sum overflow for '$first', '$row'")
                  coefficients(key) = sum
                }
              }
            case "RHS" | "RANGES" =>
              val entries = pairs(tokens)
              val sets = if (section == "RHS") rhsSets else rangeSets
              sets += first
              if (section == "RHS" && chosenRhs.isEmpty) chosenRhs = Some(first)
              if (section == "RANGES" && chosenRanges.isEmpty) chosenRanges = Some(first)
              val selected = if (section == "RHS") chosenRhs else chosenRanges
              if (selected.contains(first)) {
                val target = if (section == "RHS") rhs else ranges
                entries.foreach { case (row, value) =>
                  if (section == "RANGES" && rows(row) == "N") fail("RANGES cannot target the objective row")
                  if (target.contains(row)) fail(s"Duplicate $section value for '$row' in '$first'")
                  target(row) = value
                }
              }
            case "BOUNDS" =>
              if (tokens.size < 3 || tokens.size > 4) fail("Expected bound type, set, column, and optional value")
              val valued = Set("LO", "UP", "FX", "LI", "UI")
              if (!valued(first) && !Set("FR", "MI", "PL", "BV")(first)) fail(s"Unsupported bound type '$first'; semi-continuous and SOS domains are not supported")
              if (tokens.size != (if (valued(first)) 4 else 3)) fail(s"Wrong number of fields for $first")
              val value = tokens.lift(3).map(number)
              val b = columns.getOrElse(tokens(2), fail(s"Bound references unknown column '${tokens(2)}'"))
              boundSets += tokens(1)
              if (chosenBounds.isEmpty) chosenBounds = Some(tokens(1))
              if (chosenBounds.contains(tokens(1))) first match {
                case "LO" | "LI" => b.lower = value.get; b.explicitLower = true; if (first == "LI") b.category = Integer
                case "UP" | "UI" => b.upper = value; b.explicitUpper = true; if (first == "UI") b.category = Integer
                case "FX" => b.lower = value.get; b.upper = value; b.explicitLower = true; b.explicitUpper = true
                case "FR" => b.lower = Double.NegativeInfinity; b.upper = None; b.explicitLower = true; b.explicitUpper = true
                case "MI" => b.lower = Double.NegativeInfinity; b.explicitLower = true
                case "PL" => b.upper = None; b.explicitUpper = true
                case "BV" => b.category = Binary; b.lower = 0.0; b.upper = Some(1.0); b.explicitLower = true; b.explicitUpper = true
              }
            case _ => fail(s"Unsupported section or record '$first'")
          }
        }
        raw = reader.readLine()
      }
      if (!ended || !seenSections("ROWS") || !seenSections("COLUMNS") || objective.isEmpty)
        fail("Required ROWS, COLUMNS, objective N row or ENDATA is missing")
      if (objectiveName.exists(_ != objective)) fail(s"OBJNAME does not match the free objective row '$objective'")
      Vector((options.rhsSet, rhsSets, "RHS"), (options.boundSet, boundSets, "BOUNDS"), (options.rangeSet, rangeSets, "RANGES"))
        .foreach { case (requested, found, label) => requested.foreach(n => if (!found(n)) fail(s"Requested $label set '$n' does not exist")) }
      val model = LpProblem(modelName, options.objectiveSense.getOrElse(sense))
      val variables = columns.map { case (name, b) =>
        if (b.category == Integer && b.markerInteger && !b.explicitUpper) b.upper = Some(1.0)
        if (!b.explicitLower && b.upper.exists(_ < 0.0)) b.lower = Double.NegativeInfinity
        if (b.upper.exists(_ < b.lower)) fail(s"Inconsistent bounds for '$name'")
        name -> model.variable(name, b.lower, b.upper, b.category)
      }.toMap
      val byRow = coefficients.iterator.filter(_._2 != 0.0).toVector.groupBy(_._1._2)

      def expression(row: String): LpExpr = byRow.getOrElse(row, Vector.empty).foldLeft(LpExpr.zero) {
        case (sum, ((column, _), value)) => sum + value * variables(column)
      }

      model.setObjective(expression(objective) + -rhs.getOrElse(objective, 0.0))
      val usedNames = mutable.Set.empty[String]
      val constraints = rows.iterator.filter(_._1 != objective).map { case (name, kind) =>
        val expr = expression(name)
        val right = rhs.getOrElse(name, 0.0)
        val cs = ranges.get(name) match {
          case None => Vector((name, kind match {
            case "E" => expr === right;
            case "L" => expr <= right;
            case "G" => expr >= right
          }))
          case Some(range) =>
            val magnitude = math.abs(range)
            val lower = if (kind == "G" || kind == "E" && range >= 0.0) right else right - magnitude
            val upper = if (kind == "L" || kind == "E" && range < 0.0) right else right + magnitude
            if (!java.lang.Double.isFinite(lower) || !java.lang.Double.isFinite(upper)) fail(s"Range overflow for '$name'")
            Vector((s"${name}__lower", expr >= lower), (s"${name}__upper", expr <= upper))
        }
        name -> cs.map { case (n, c) =>
          if (usedNames(n) || n != name && rows.contains(n)) fail(s"Expanded range name '$n' collides with another row")
          usedNames += n
          val named = c.named(n);
          model += named;
          named
        }
      }.toMap
      LpMpsModel(model, variables, constraints, chosenRhs, chosenBounds, chosenRanges)
    } finally reader.close()
  }
}
