# Test layers

Keep arithmetic and policy checks in Spark-free suites. Use Spark integration
suites for variable expansion, numerical solves, callbacks, and cache ownership.
Test the same domain interfaces in both layers; avoid replacing solver checks
with mocks of solver internals.

The arithmetic, MIP-gap, and solve-control suites follow this split. Their test
names and assertions are preserved. Other existing suites keep their current
structure.

Run these domain checks without starting Spark:

```sh
sbt 'spark-lpSpark_3_52_12/testOnly *LpArithmeticSuite *LpMipGapSuite *SolveControlSuite'
```

Run their Spark integration checks:

```sh
sbt 'spark-lpSpark_3_52_12/testOnly *LpArithmeticIntegrationSuite *LpMipGapIntegrationSuite *SolveControlIntegrationSuite'
```

Run the complete library suite, including both layers:

```sh
sbt 'spark-lpSpark_3_52_12/test'
```

These commands select Spark 3.5. Use the corresponding project identifier for
another supported Spark version. Use a Java version supported by that Spark
version; the commands above are validated with Java 11.
