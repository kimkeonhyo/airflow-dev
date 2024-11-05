from airflow import DAG
import pendulum
import datetime
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
import glob

with DAG(
    dag_id="03-python-operator",
    schedule="30 18 * * *",
    start_date=pendulum.datetime(2024, 10, 30, tz="Asia/Seoul"),
    catchup=False
) as dag:
    def merge_logs(format: str):
        log_files_list = glob.glob(f"data/logs/*.{format}")
        merged_log_contents = ""
        for file in log_files_list:
            with open(file, "r") as f:
                for line in f:
                    merged_log_contents += line
                    
        return merged_log_contents # XCom으로 push됨
    
    def send_email(**kwagrs):
        merged_log_contents = kwagrs['ti'].xcom_pull(task_ids='merge_log_task')
        # ti: task_instance
        
        email_operator = EmailOperator(
            task_id='send_email_task', # task_id와 변수명 일치시키기
            to='rjsgy033@gmail.com',
            subject='[Airflow] ✅ Success !',
            html_content=f' DAG: {{ task_instance.dag_id }}<br> Task: {{ task_instance.task_id }}<br> Execution Time: {{ ts }}<br> Log URL: {{ task_instance.log_url }}<br> log: { merged_log_contents }'
            # cc="" # 참조
        )
        email_operator.execute(context=kwagrs)

    merge_log_task = PythonOperator(
        task_id='merge_log_task',
        python_callable=merge_logs,
        op_kwargs={'format':'txt'},
    )
    
    send_email_task = PythonOperator(
        task_id='send_email_task',
        python_callable=send_email,
        provide_context=True,  # XCom을 사용하기 위해 필요
    )

    merge_log_task >> send_email_task