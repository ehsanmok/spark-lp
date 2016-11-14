name := "spark-lp"

version := "1.0-SNAPSHOT"

scalaVersion := "2.10.4"

sparkVersion := "2.0.0"

sparkComponents += "mllib"

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

libraryDependencies ++= Seq(
  "com.joptimizer" % "joptimizer" % "3.4.0",
  "org.scalatest" %% "scalatest" % "2.1.5" % Test
)

parallelExecution in Test := false

// META-INF discarding
mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
   {
    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
    case x => MergeStrategy.last
   }
}
