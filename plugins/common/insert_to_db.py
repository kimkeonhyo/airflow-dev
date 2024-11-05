import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy.types import Integer, String, DateTime, VARCHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd

TABLE_NAME = "logs_table_created_by_dohyung"

def create_and_insert_to_db(**kwargs): # ✅
    # ✅
    ti = kwargs['ti']
    df_json = ti.xcom_pull(task_ids="preprocess_logs_task")
    df = pd.read_json(df_json, orient="split")
    
    load_dotenv(dotenv_path='.env')
    postgresql_connection_info = os.getenv("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
    postgresql_connection_info += '/postgres'

    engine = create_engine(
        url=postgresql_connection_info,
        echo=True,
    )

    Base = declarative_base()
    
    class LogsTable(Base):
        __tablename__ = TABLE_NAME
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        IP = Column(VARCHAR)
        timestamp = Column(DateTime)
        http_method = Column(VARCHAR)
        endpoint = Column(VARCHAR)
        status_code = Column(VARCHAR)
        
    inspector = inspect(engine)
    
    if not inspector.has_table(TABLE_NAME):
        Base.metadata.create_all(engine, tables=[LogsTable.__table__])
        print(f"{TABLE_NAME} 테이블이 생성되었습니다.")
    else:
        print(f"{TABLE_NAME} 테이블이 이미 존재합니다.")
    
    # ✅
    # import read_logs
    # df = read_logs.read_log()
    
    sessionObject = sessionmaker(bind=engine)
    try:
        with sessionObject() as session:
            # DataFrame을 딕셔너리 리스트로 변환
            records = df.to_dict(orient='records')
            
            # Bulk insert
            session.bulk_insert_mappings(LogsTable, records)
            session.commit()
            print(f"{len(records)}개의 레코드가 {TABLE_NAME} 테이블에 삽입되었습니다.")
    except Exception as e:
        print(f"데이터 삽입 중 오류가 발생했습니다: {e}")