from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import round as sparkRound, col, to_timestamp
from pyspark.sql.functions import max as spMax
from pyspark.sql.functions import min as spMin
from pyspark.sql.functions import avg as spAvg
from pyspark.sql.functions import hour, minute, date_trunc, collect_set, count


def load_data(ss: SparkSession, schema: StructType) -> DataFrame:
    return ss.read.schema(schema).text("data/spark/log.txt")


def kb_converted_size(size_bytes):
    # print(type(size_bytes))
    return sparkRound((size_bytes / 1024), 2)


if __name__ == '__main__':
    ss: SparkSession = SparkSession.builder.master("local").appName("DataFrame").getOrCreate()

    # 스키마 정의
    schema = StructType(fields=[
        StructField(name="ip", dataType=StringType(), nullable=False),
        StructField(name="timestamp", dataType=StringType(), nullable=False),
        StructField(name="http_method", dataType=StringType(), nullable=False),
        StructField(name="endpoint", dataType=StringType(), nullable=False),
        StructField(name="status_code", dataType=StringType(), nullable=False),
        StructField(name="bytes", dataType=StringType(), nullable=False)
    ])

    df = load_data(ss, schema)
    # df.show()
    # df.printSchema()

    # bytes 단위를 KB 단위로 변환하기
    # df = df.withColumn("KB", kb_converted_size(df.bytes))
    df = df.withColumn("KB", kb_converted_size(col("bytes")))
    # df.show()
    # df.printSchema()

    # timestamp 열의 타입 변경
    df = df.withColumn("timestamp", to_timestamp(col("timestamp"), "yyyy-MMM-dd HH:mm:ss"))

    # /users 엔드포인트에서 상태코드가 400인 행을 filtering
    users_400_df = df.filter((df.endpoint == "/users") & (df.status_code == "400"))
    # users_400_df.show()

    # groupby> 엔드포인트별 각 HTTP 메서드의 bytes의 최대, 최소, 평균 구하기
    group_1_df = df.groupby(["endpoint", "status_code"]).agg(
        spMax("bytes").alias("max_bytes"),
        spMin("bytes").alias("min_bytes"),
        spAvg("bytes").alias("avg_bytes"),
    )
    # group_1_df.show()

    df.show()

    # groupby> 분 단위, 중복 제거 / IP 리스트 및 개수 추출
    group_columns = ["hour", "minute"]
    group_2_df = df.withColumn(
        "hour", hour(date_trunc(format="hour", timestamp=col("timestamp")))
    ).withColumn(
        "minute", minute(date_trunc(format="minute", timestamp=col("timestamp")))
    ).groupby(group_columns).agg(
        collect_set("ip").alias("ip_list"),
        count("ip").alias("ip_count")
    ).sort(group_columns)
    group_2_df.show()


