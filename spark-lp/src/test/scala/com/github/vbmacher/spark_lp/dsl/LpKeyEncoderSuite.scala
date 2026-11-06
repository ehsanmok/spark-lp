package com.github.vbmacher.spark_lp.dsl

import org.scalatest.funsuite.AnyFunSuite

class LpKeyEncoderSuite extends AnyFunSuite {
  test("type tags distinguish strings, decimals and booleans while canonicalizing integral types") {
    val parts: Seq[Any] = Seq("1", 1, 1L, 1.0, true)

    val encoded = parts.map(part => KeyCodec.encodeParts(Seq(part)))

    assert(encoded == Seq("s:1", "i:1", "i:1", "d:3ff0000000000000", "b:true"))
  }

  test("exact float values share double encodings while inexact float values remain distinct") {
    val exactlyRepresentable = 1.5
    val inexactDecimal = 0.1

    val doubleExact = KeyCodec.encodePart(exactlyRepresentable)
    val floatExact = KeyCodec.encodePart(exactlyRepresentable.toFloat)
    val doubleInexact = KeyCodec.encodePart(inexactDecimal)
    val floatInexact = KeyCodec.encodePart(inexactDecimal.toFloat)

    assert(doubleExact == "d:3ff8000000000000")
    assert(floatExact == doubleExact)
    assert(doubleInexact != floatInexact)
  }

  test("multi-part keys use the unit separator without colliding with strings") {
    val parts: Seq[Any] = Seq("a", 1)

    val composite = KeyCodec.encodeParts(parts)
    val concatenated = KeyCodec.encodeParts(Seq("a1"))
    val stringParts = KeyCodec.encodeParts(Seq("a", "1"))

    assert(composite == "s:a\u001Fi:1")
    assert(concatenated != composite)
    assert(stringParts != composite)
  }

  test("repeated encoding preserves the complete canonical key") {
    val parts: Seq[Any] = Seq("beef", 42, 1.25, true)

    val firstEncoding = KeyCodec.encodeParts(parts)
    val secondEncoding = KeyCodec.encodeParts(parts)

    assert(firstEncoding == "s:beef\u001Fi:42\u001Fd:3ff4000000000000\u001Fb:true")
    assert(secondEncoding == firstEncoding)
  }

  test("a null part or empty parts produce a null key") {
    val withNull: Seq[Any] = Seq("a", null)
    val onlyNull: Seq[Any] = Seq(null)
    val noParts = Seq.empty[Any]

    val mixed = KeyCodec.encodeParts(withNull)
    val single = KeyCodec.encodeParts(onlyNull)
    val scalar = KeyCodec.encodeValue(null)
    val empty = KeyCodec.encodeParts(noParts)

    assert(mixed == null)
    assert(single == null)
    assert(scalar == null)
    assert(empty == null)
  }

  test("built-in encoders flatten scalar and tuple keys in declaration order") {
    val scalarEncoder = implicitly[LpKeyEncoder[String]]
    val pairEncoder = implicitly[LpKeyEncoder[(String, Int)]]
    val tripleEncoder = implicitly[LpKeyEncoder[(String, Int, Boolean)]]
    val quadrupleEncoder = implicitly[LpKeyEncoder[(String, Int, Boolean, Double)]]

    val scalar = scalarEncoder.parts("x")
    val pair = pairEncoder.parts(("x", 3))
    val triple = tripleEncoder.parts(("x", 3, true))
    val quadruple = quadrupleEncoder.parts(("x", 3, true, 0.5))

    assert(scalar == Seq("x"))
    assert(pair == Seq("x", 3))
    assert(triple == Seq("x", 3, true))
    assert(quadruple == Seq("x", 3, true, 0.5))
  }

  test("a caller-defined encoder exposes components used by canonical key encoding") {
    case class PlantProduct(plant: String, product: Int)
    val key = PlantProduct("p1", 7)
    val encoder: LpKeyEncoder[PlantProduct] = LpKeyEncoder.instance(k => Seq(k.plant, k.product))

    val parts = encoder.parts(key)
    val encoded = KeyCodec.encodeParts(parts)

    assert(parts == Seq("p1", 7))
    assert(encoded == "s:p1\u001Fi:7")
  }

  test("display names join parts and escape commas, brackets and backslashes") {
    val escapedParts = Seq("a,b", "c[d]", "e\\f")

    val single = KeyCodec.displayName("amount", Seq("beef"))
    val composite = KeyCodec.displayName("amount", Seq("a", "b"))
    val escaped = KeyCodec.displayName("v", escapedParts)
    val scalar = KeyCodec.displayName("scalar", Seq.empty)

    assert(single == "amount[beef]")
    assert(composite == "amount[a,b]")
    assert(escaped == "v[a\\,b,c\\[d\\],e\\\\f]")
    assert(scalar == "scalar")
  }
}
