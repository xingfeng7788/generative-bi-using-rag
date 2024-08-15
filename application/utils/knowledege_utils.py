import os

import requests
from retrying import retry

from utils.superset_conn import create_admin_token


@retry(stop_max_attempt_number=3, wait_fixed=1000)
def get_knowledege_context(database_id, schema, table_name, database_engine):
    if database_engine.lower() == "clickhouse":
        dialect = "ClickHouse"
    elif database_engine.lower() == "redshift":
        dialect = "RedShift"
    else:
        dialect = database_engine.capitalize()
    token = create_admin_token()
    url = f"{os.getenv('VALIDATE_HOST')}/dops-dataopt/dataset/detailInfo?key={dialect}/{database_id}://datalake-cluster.{schema}/{table_name}"
    headers = {
        'authorization': token,
        'Content-Type': 'application/json'
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception("500 Internal Server Error: Failed to get knowledge context")
    else:
        res = response.json()['data']
        return res
