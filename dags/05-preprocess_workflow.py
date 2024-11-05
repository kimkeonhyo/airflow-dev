from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import pendulum
import logging
from common.read_logs import read_log
from common.preprocess_logs import preprocess
from common.insert_to_db import create_and_insert_to_db
# Airflow는 plugins 폴더까지를 경로로 잡고 있기 때문에 common부터 명시
# .env 파일에 PYHTONPATH 명시하면 import 에러가 사라짐

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with DAG(
    dag_id="05-preprocess_workflow",
    schedule="0 9 * * *",
    start_date=pendulum.datetime(2024, 10, 31, tz="Asia/Seoul"),
    catchup=False
) as dag:
    start_task = EmptyOperator(task_id="start_task")
    end_task = EmptyOperator(task_id="end_task")
    
    read_logs_task = PythonOperator(
        task_id="read_logs_task",
        python_callable=read_log,
        provide_context=True
    )
    preprocess_logs_task = PythonOperator(
        task_id="preprocess_logs_task",
        python_callable=preprocess,
        provide_context=True
    )
    insert_logs_to_db_task = PythonOperator(
        task_id="insert_logs_to_db_task",
        python_callable=create_and_insert_to_db,
        provide_context=True
    )
    
    start_task >> read_logs_task >> preprocess_logs_task >> insert_logs_to_db_task >> end_task