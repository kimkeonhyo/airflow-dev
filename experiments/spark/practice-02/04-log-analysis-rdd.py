from datetime import datetime
from typing import List

from pyspark.sql import SparkSession
from pyspark import SparkContext, RDD

if __name__ == '__main__':
    ss: SparkSession = SparkSession.builder\
        .master("local")\
        .appName("Log Analysis using RDD")\
        .getOrCreate()
    # Session 불러오기

    # RDD 사용하기: Context 객체 필요
    sc: SparkContext = ss.sparkContext

    log_analysis_rdd: RDD[str] = sc.textFile("data/spark/log.txt")

    # count 세기
    print(f"count of RDD: {log_analysis_rdd.count()}")

    # 각 row 출력
    log_analysis_rdd.foreach((lambda  v: print(v)))

    # transformation 1: map
    # log 파일의 각 행을 list의 str 타입으로 불러오는 map함수 만들기
    def parser(row: str):
        return row.strip().split(" | ")

    parsed_rdd: RDD[List[str]] = log_analysis_rdd.map(parser)

    parsed_rdd.foreach(lambda v: print(v))

    # transformation 2: filter
    # status code가 404인 로그만 필터링하는 연산
    # parsed_rdd 리스트 기준으로 맨 마지막 요소에 해당함
    # 마지막 값을 가져와서 404인지 체크
    def get_404(row: List[str]) -> bool:
        status_code = row[-1]
        return status_code == '404'

    status_code_rdd = parsed_rdd.filter(get_404)

    status_code_rdd.foreach(print)

    # status code가 정상인 로그만 필터링하기
    def get_202(row: List[str]) -> bool:
        status_code = row[-1]
        return status_code == "200"

    status_code_rdd = parsed_rdd.filter(get_202)

    status_code_rdd.foreach(print)

    # POST 요청이고, /customers 끝나는 로그만 필터링하기
    def get_post_request_and_customers_api(row: List[str]) -> bool:
        # 따옴표 제거
        log = row[2].replace("\"", "")
        return log.startswith("POST") and "/customer" in log

    customers_post_request_rdd = parsed_rdd.filter(get_post_request_and_customers_api)

    customers_post_request_rdd.foreach(print)

    # transformation 3: Reduce 연산
    # RESTful API 메서드의 종류에는 POST, GET, PUT, PATCH, DELETE가 있다.
    # 각 API별 개수 카운팅
    def extract_api(row: List[str]) -> tuple[str, int]:
        log = row[2].replace("\"", "")
        api = log.split(" ")[0]
        # 리턴 목표: 튜플 형태(key, value)
        return api, 1

    count_of_api_rdd = parsed_rdd.map(extract_api).reduceByKey(lambda c1, c2: c1 + c2).sortByKey()

    count_of_api_rdd.foreach(print)

    # 2-시간 및 분 단위별 요청 횟수를 출력하기
    def extract_hours_and_minutes(row: List[str]) -> tuple[str, int]:
        timestamps = row[1].replace("[", "").replace("]", "")
        date_format = "%d/%b/%Y:%H:%M:%S" # data format이 어떤지 먼저 명시
        date = datetime.strptime(timestamps, date_format) # datetime 객체 생성하기
        return f"{date.hour}:{date.minute}", 1

    hours_and_minutes_rdd = parsed_rdd.map(
        extract_hours_and_minutes
    ).reduceByKey(
        lambda c1, c2 : c1 + c2
    ).sortByKey()

    hours_and_minutes_rdd.foreach(print)

    # transformation 4 - groupby 연산
    # 1> status code, api 메서드별 IP 리스트 출력하기
    def extract_cols(row: List[str]) -> tuple[str, str, str]:
        ip = row[0]
        status_code = row[-1]
        api_method = row[2].replace("\"", "").split(" ")[0]

        return status_code, api_method, ip

    result = parsed_rdd.map(
        extract_cols
        # 하나의 키가 아닌 여러 개의 키를 하나의 키로 묶어야 하기 때문에 map을 여러 개 써야 한다.
    ).map(
        lambda x: ((x[0], x[1]), x[2])).groupByKey().mapValues(list)

    result.foreach(print)

    # groupbykey가 아닌 reduceByKey 이용하여 구현하기
    result = parsed_rdd.map(
        extract_cols
    ).map(
        lambda x: ((x[0], x[1]), x[2])
    ).reduceByKey( # str으로 하나로 묶기
        # shuffle이 발생하여, stage로 나누어진다.
        lambda i1, i2: f"{i1},{i2}"
    ).map(
        lambda row: (row[0], row[1].split(",")) # list로 만들어주기
    )

    result.foreach(print)

    while True:
        pass