from airflow import DAG
import pendulum
from datetime import datetime
from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator

def print_name_time(**kwargs):  # kwargs를 사용하여 context 정보를 받음
    now = datetime.now().strftime('%Y.%m.%d - %H:%M:%S')
    dag_name = kwargs['task_instance'].dag_id
    print(f"DAG name: {dag_name}")
    print(f"Current time: {now}")
    return "success"  # 성공적으로 실행된 경우 반환

with DAG(
    dag_id="03-printname-operator", # 파일 명과 일치시키기
    schedule="0 13 * * *", # 매월 1일 오전 9시
    start_date=pendulum.datetime(2024, 11, 3, tz="Asia/Seoul"),
    catchup=False
) as dag:
    
    print_name_task = PythonOperator(
        task_id='print_name_task',
        python_callable=print_name_time,
        provide_context=True
    )
    send_email_task = EmailOperator(
        task_id='send_email_task', # task_id와 변수명 일치시키기
        to='rjsgy033@gmail.com',
        subject='[Airflow] ✅ Success !',
        html_content=' DAG: {{ task_instance.dag_id }}<br> Task: {{ task_instance.task_id }}<br> Execution Time: {{ ts }}<br> Log URL: {{ task_instance.log_url }}'
        # Fail mail send 못함 계속해서 생각해볼 것 
        # branchOperator로 task 시작 순서 제어
        # python 함수를 이용하여 task 시작 조건 제시
    )

print_name_task >> send_email_task