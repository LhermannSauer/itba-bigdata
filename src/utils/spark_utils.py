from pyspark.sql import SparkSession

def get_spark(app_name: str = "LocalSparkApp") -> SparkSession:
    spark = (
        SparkSession.builder
         .appname(app_name)
         .master('local[*]')
         .config('spark.sql.shuffle.partitions','4')
         .config('spark.driver.memory','4g')
         .config('spark.sql.execution.arrow.pyspark.enabled','true')
         .getOrCreate()
    )
    return spark
