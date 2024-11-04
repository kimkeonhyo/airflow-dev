from airflow import DAG
import pendulum
import datetime
from airflow.operators.email import EmailOperator

with DAG(
    dag_id="02-email-operator", # 파일 명과 일치시키기
    schedule="0 9 1 * *", # 매월 1일 오전 9시
    start_date=pendulum.datetime(2024, 10, 30, tz="Asia/Seoul"),
    catchup=False
) as dag:
    send_email_task = EmailOperator(
        task_id='send_email_task', # task_id와 변수명 일치시키기
        to='rjsgy033@gmail.com',
        subject='[Airflow] ✅ Success !',
        html_content=' DAG: {{ task_instance.dag_id }}<br> Task: {{ task_instance.task_id }}<br> Execution Time: {{ ts }}<br> Log URL: {{ task_instance.log_url }}'
        # cc="" # 참조
    )