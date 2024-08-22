# -*- coding: UTF-8 -*-
# @Project ：generative-bi-using-rag 
# @FileName ：public_utils.py
# @Author ：dingtianlu
# @Date ：2024/8/19 13:46
# @Function  :
import json
import os
import sqlalchemy as db
from api.enum import ContentEnum
from fastapi import WebSocket
from utils.logging import getLogger
logger = getLogger()


def execute_sql(sql):
    engine = db.create_engine(os.getenv("SUPERSET_DATABASE_URI"))
    with engine.connect() as con:
        rs = con.execute(sql)
        data = [dict(zip(result.keys(), result)) for result in rs]
    return data


async def deal_response(content, content_type, session_id, status, user_id):
    if content_type == ContentEnum.STATE:
        content_json = {
            "text": content,
            "status": status
        }
        content = content_json
    content_obj = {
        "session_id": session_id,
        "user_id": user_id,
        "content_type": content_type.value,
        "content": content,
    }
    logger.info(content_obj)
    final_content = json.dumps(content_obj, ensure_ascii=False)
    return final_content


async def response_stream(session_id: str, content,
                          content_type: ContentEnum = ContentEnum.COMMON, status: str = "-1",
                          user_id: str = "admin"):
    final_content = await deal_response(content, content_type, session_id, status, user_id)
    yield f"data: {final_content}\n\n"


async def response_websocket(websocket: WebSocket, session_id: str, content,
                             content_type: ContentEnum = ContentEnum.COMMON, status: str = "-1",
                             user_id: str = "admin"):
    final_content = await deal_response(content, content_type, session_id, status, user_id)
    await websocket.send_text(final_content)


async def response_stream_json(session_id: str, content,
                               content_type: ContentEnum = ContentEnum.COMMON, status: str = "-1",
                               user_id: str = "admin", streamed_json: bool = False):
    final_content = await deal_response(content, content_type, session_id, status, user_id)
    if streamed_json:
        yield f"data: {final_content}\n\n"
    else:
        yield final_content
