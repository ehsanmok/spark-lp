package com.github.vbmacher.spark_lp

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.storage.StorageLevel
import org.scalatest.funsuite.AnyFunSuite

class CachedRDDsSuite extends AnyFunSuite with DataFrameSuiteBase {

  test("checkpointed values survive cache loss without recomputing their source") {
    val caches = new CachedRDDs
    val evaluations = sc.longAccumulator("source evaluations")
    val values = caches.checkpoint(sc.parallelize(1 to 8, 2).map { value =>
      evaluations.add(1L)
      value * 2
    })
    try {
      val expected = (1 to 8).map(_ * 2).toArray

      assert(values.collect().sameElements(expected))
      assert(values.getCheckpointFile.nonEmpty)
      val before = evaluations.value

      values.unpersist(blocking = true)

      assert(values.collect().sameElements(expected))
      assert(evaluations.value == before)
    } finally caches.close()
  }

  test("checkpoint ownership preserves caller caches and releases owned caches") {
    val caches = new CachedRDDs
    val caller = sc.parallelize(1 to 4, 2).persist(StorageLevel.DISK_ONLY)
    try {
      caches.checkpoint(caller).count()
      val owned = caches.checkpoint(caller.map(_ * 2))
      owned.count()
      caches.close()

      assert(caller.getStorageLevel == StorageLevel.DISK_ONLY)
      assert(owned.getStorageLevel == StorageLevel.NONE)
      assert(owned.collect().sameElements(Array(2, 4, 6, 8)))
    } finally {
      caches.close()
      caller.unpersist(blocking = true)
    }
  }
}
