from pyspark.sql import SparkSession
import pyspark.sql.functions as f

if __name__ == "__main__":
    # Spark 프레임워크를 사용하기 위한 진입점 정의
    ss: SparkSession = SparkSession.builder \
        .master("local") \
        .appName("wordCount using RDD") \
        .getOrCreate()
    # 세션 객체 생성됨

    # 이전 예제와는 다르게, SparkContext 없이 SparkSession에서 작업
    df = ss.read.text("data/spark/LoremIpsum.txt")
    # df: dataframe. RDD 아님. RDD보다 더 고차원/고수준의 자료구조. pandas와 유사함

    # transformation
    value = f.col('value')
    splitted_value = f.split(value, pattern=" ")
    flatten_value = f.explode(splitted_value) # f.expldoe: flatMap처럼 한 줄에 있던 것을 여러 줄로 평탄화한다.
    value_df = df.withColumn('word', flatten_value)

    mapped_value = f.lit(1)
    mapped_df = value_df.withColumn("count", mapped_value)

    word_count_df = mapped_df.groupby("word").sum()

    word_count_df.show()