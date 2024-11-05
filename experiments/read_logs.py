import pandas as pd

def read_log():
    print(f"pandas version: \n{pd.__version__}\n\n")
    
    # pandas로 데이터 불러오기
    df = pd.read_csv(filepath_or_buffer="data/logs/log-241001.txt", delimiter=" ")
    print(f"df: \n{df}\n\n")
    
    # 불러온 데이터의 모습 살펴보기
    desctiption_df = df.describe()
    print(f"description about df: \n{desctiption_df}\n\n")
    
    # 불러온 데이터의 column 확인하기
    columns_df = df.columns
    print(f"columns: \n{columns_df}\n\n")
    # 컬럼의 이름이 지정되지 않은 모습이 보인다.
    
    # 컬럼의 이름을 지정하기 전에, 컬럼의 개수 확인하기
    length_column_of_df = df.shape[1]
    print(f"length_column_of_df: \n{length_column_of_df}\n\n")
    # 7개
    
    # 컬럼에 이름 부여하고, 확인하기
    df.columns = ['IP', "drop", "date_and_time", "drop2", "http_methods_and_endpoint", "drop3", "status_code"]
    print(f"df: \n{df}\n\n")
    columns_df = df.columns
    print(f"columns: \n{columns_df}\n\n")
    
    # 불필요한 컬럼 제거하기(되는지 안되는지 TEST 용)
    # _for_check = df.drop(labels=["drop", "drop2", "drop3"], axis=1, inplace=False)
    # print(f"_for_check: {_for_check}")
    truncated_df = df.drop(labels=["drop", "drop2", "drop3"], axis=1, inplace=False)
    
    # checkpoint
        # http_methods_and_endpoint
            # http_methods와 endpoint를 분리해야 함
                # ex. PATCH /events HTTP/1.1 ---> PATCH 와 /events 로
        # date_and_time에서 괄호 제거
        # 각 컬럼별로 타입이 적절한가?
            # IP: string
            # date_and_time: timestamp
            # http_methods: string
            # endpoint: string
            # status_code: string

    # http_method와 endpoint를 추출
    print(f"truncated_df.http_methods_and_endpoint: \n{truncated_df.http_methods_and_endpoint}\n\n")
    http_method_and_endpoint = truncated_df.http_methods_and_endpoint
    http_method = http_method_and_endpoint.map(lambda x: x.split(" ")[0])
    endpoint = http_method_and_endpoint.map(lambda x: x.split(" ")[1])
    print(f"http_method: \n{http_method}\n\n")
    print(f"endpoint: \n{endpoint}\n\n")
    
    # 기존 컬럼 제거
    truncated_df.drop(labels="http_methods_and_endpoint", axis=1, inplace=True)
    print(f"truncated_df: \n{truncated_df}\n\n")
    
    # 새로운 컬럼 추가
    # truncated_df.merge()
    # merge를 쓰려면, merge하려는 데이터들에 index가 적절히 설정되어져 있어야 할 것 같다.
    print(f"truncated_df.index: \n{truncated_df.index}\n\n")
    print(f"http_method.index: \n{http_method.index}\n\n")
    print(f"endpoint.index: \n{endpoint.index}\n\n")
    
    # print(truncated_df.merge(http_method))
    # pandas.errors.MergeError: No common columns to perform merge on. Merge options: left_on=None, right_on=None, left_index=False, right_index=False
    # 공식문서를 봐야겠다.
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
    # merge가 아닌가?
    merged_http_method_and_endpoint = pd.concat(objs=[http_method, endpoint], axis=1, ignore_index=True)
    print(merged_http_method_and_endpoint)
    merged_http_method_and_endpoint.columns = ["http_method", "endpoint"]
    print(merged_http_method_and_endpoint)
    
    merge_df = pd.concat(objs=[truncated_df, merged_http_method_and_endpoint], axis=1, ignore_index=True)
    print(f"merge_df: \n{merge_df}\n\n")
    merge_df.columns = ["IP", "timestamp", "status_code", "http_method", "endpoint"]
    print(f"merge_df: \n{merge_df}\n\n")
    
    # timestamp에서 괄호 제거
    # print(merge_df['timestamp'].str.strip('[]'))
    merge_df['timestamp'] = merge_df['timestamp'].str.strip(to_strip='[]')
    print(f"merge_df: \n{merge_df}\n\n")
    
    # 타입 변경
    # print(pd.to_datetime(merge_df['timestamp'], format='%d/%b/%Y:%H:%M:%S'))
    merge_df['timestamp'] = pd.to_datetime(arg=merge_df['timestamp'], format='%d/%b/%Y:%H:%M:%S')
    print(f"merge_df: \n{merge_df}\n\n")
    
    # 나머지 열을 문자열 타입으로 변환
    merge_df = merge_df.astype({
        'IP': 'string',
        'http_method': 'string',
        'endpoint': 'string',
        'status_code': 'string'
    })
    print(f"merge_df.dtypes: {merge_df.dtypes}")
    
    return merge_df

if __name__ == "__main__":
    df = read_log()