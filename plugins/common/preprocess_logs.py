import pandas as pd

def preprocess(**kwargs): # ✅
    # ✅
    ti = kwargs['ti']
    df_json = ti.xcom_pull(task_ids='read_logs_task')
    df = pd.read_json(df_json, orient="split")
    
    df.columns = ['IP', "drop", "date_and_time", "drop2", "http_methods_and_endpoint", "drop3", "status_code"]
    
    truncated_df = df.drop(labels=["drop", "drop2", "drop3"], axis=1, inplace=False)

    http_method_and_endpoint = truncated_df.http_methods_and_endpoint
    http_method = http_method_and_endpoint.map(lambda x: x.split(" ")[0])
    endpoint = http_method_and_endpoint.map(lambda x: x.split(" ")[1])
    
    truncated_df.drop(labels="http_methods_and_endpoint", axis=1, inplace=True)
    
    merged_http_method_and_endpoint = pd.concat(objs=[http_method, endpoint], axis=1, ignore_index=True)
    merged_http_method_and_endpoint.columns = ["http_method", "endpoint"]
    
    merge_df = pd.concat(objs=[truncated_df, merged_http_method_and_endpoint], axis=1, ignore_index=True)
    merge_df.columns = ["IP", "timestamp", "status_code", "http_method", "endpoint"]
    
    merge_df['timestamp'] = merge_df['timestamp'].str.strip(to_strip='[]')
    merge_df['timestamp'] = pd.to_datetime(arg=merge_df['timestamp'], format='%d/%b/%Y:%H:%M:%S')
    
    merge_df = merge_df.astype({
        'IP': 'string',
        'http_method': 'string',
        'endpoint': 'string',
        'status_code': 'string'
    })
    
    print(f"merge_df.dtypes: {merge_df.dtypes}")
    
    merge_df = merge_df.to_json(orient="split")
    
    return merge_df