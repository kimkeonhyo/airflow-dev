from airflow import DAG
import pendulum
import datetime
from airflow.operators.empty import EmptyOperator

with DAG(
    dag_id="01-empty-operator",
    schedule="10 0 * * 6#1", # 매월 첫 번째 주 토요일 0시 10분
    start_date=pendulum.datetime(2024, 10, 29, tz="Asia/Seoul"),
    catchup=False
) as dag:
    
    t1 = EmptyOperator(
        task_id="t1"
    )

    t2 = EmptyOperator(
        task_id="t2"
    )

    t3 = EmptyOperator(
        task_id="t3"
    )

    t4 = EmptyOperator(
        task_id="t4"
    )

    t5 = EmptyOperator(
        task_id="t5"
    )

    t6 = EmptyOperator(
        task_id="t6"
    )

    t7 = EmptyOperator(
        task_id="t7"
    )

    t8 = EmptyOperator(
        task_id="t8"
    )

    t1 >> [t2, t3] >> t4
    t5 >> t4 
    [t4, t7] >> t6 >> t8