package com.github.vbmacher.spark_lp

import org.scalatest.funsuite.AnyFunSuite

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration._

class SolveControlSuite extends AnyFunSuite {
  test("default controls are stable and monitor stamps phase and metric events") {
    assert(SolveControl() == SolveControl(), "default callbacks must not make an integer model controlled")
    val events = ArrayBuffer.empty[SolveProgress]
    var now = 0L
    val monitor = new SolveMonitor(SolveControl(onProgress = events += _), () => now)
    monitor.phase(SolvePhase.Initialization)
    monitor.iteration = 3
    now = 2000000000L
    monitor.report(SolvePhase.SystemSetup, work = Some(WorkProgress(2, Some(4))))

    assert(events.head.iteration == 0)
    assert(events.head.iterate.isEmpty)
    assert(events.head.work.isEmpty)
    assert(events.last.iteration == 3)
    assert(events.last.elapsedSeconds == 2.0)
    assert(events.last.work.contains(WorkProgress(2, Some(4))))
    assert(events.last.iterate.isEmpty)
  }

  test("deadline uses an injected monotonic clock; no sleeps") {
    var now = 100L
    val monitor = new SolveMonitor(SolveControl(timeLimit = Some(10.nanos)), () => now)
    now = 109L
    monitor.check()
    now = 110L

    assert(intercept[SolveStopped](monitor.check()).reason == StopReason.TimeLimit)
  }

  test("progress windows tolerate regressions, accumulate gains, and stop on noise") {
    val config = StagnationConfig(patience = 3, absoluteImprovement = 0.1, relativeImprovement = 0.0)
    val window = new ProgressWindow(config, 3)
    Seq(10.0, 9.0, 9.5, 8.5, 8.8, 8.45).zipWithIndex.foreach { case (v, i) =>
      assert(!window.observe(i, Vector(v)))
    }

    assert(window.observe(6, Vector(8.49)))
    val gradual = new ProgressWindow(config, 3)
    (0 to 20).foreach(i => assert(!gradual.observe(i, Vector(10.0 - i * 0.06))))
  }

  test("flat objective with improving residuals continues; feasibility is not repeatedly reset") {
    val policy = new OuterProgress(StagnationConfig(patience = 2))
    (1 to 10).foreach(i => assert(!policy.observe(i, true, 5.0, 0.0, 0.0, 1.0 / i, 1.0 / i)))

    assert(!policy.observe(11, false, 0.0, 1.0, 1.0, 1.0, 1.0))
    assert(policy.observe(12, true, 5.0, 0.0, 0.0, 0.1, 0.1))
  }

  test("inner progress window spans rank retries and uses cumulative steps") {
    val policy = new ProgressWindow(StagnationConfig(relativeImprovement = 0.1), 100)

    assert(!policy.observe(0, Vector(1.0)))
    assert(!policy.observe(50, Vector(0.8)))
    assert(!policy.observe(100, Vector(0.79)))
    assert(policy.observe(150, Vector(0.78)))
  }
}
