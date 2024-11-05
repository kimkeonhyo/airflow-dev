from pyspark import SparkContext, RDD
from pyspark.sql import SparkSession


if __name__ == '__main__':
    # Spark 프레임워크를 사용하기 위한 진입점 정의
    ss: SparkSession = SparkSession.builder\
                        .master("local")\
                        .appName("wordCount using RDD")\
                        .getOrCreate()
    # 세션 객체 생성됨

    # RDD 자료구조를 정의하기 위해 컨텍스트로부터 스파크 컨텍스트 추출
    sc: SparkContext = ss.sparkContext

    # 컨텍스트를 이용하여 데이터 Load
    data: RDD[str] = sc.textFile("data/spark/LoremIpsum.txt")

    # 불러온 데이터를 처리를 위해 Transformation
    counts = data\
        .flatMap(lambda line: line.split())\
        .map(lambda word: (word, 1))\
        .reduceByKey(lambda count_1, count_2: count_1 + count_2)

    print(f"counts before action: {counts}")

    # Action
    output = counts.collect()

    print(f"counts after action")
    for word, count in output:
        print(f"{word}: {count}")