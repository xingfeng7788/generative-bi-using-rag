# -*- coding:utf-8 -*-
# @FileName  : chart.py
# @Time      : 2024/7/19 14:53
# @Author    : dingtianlu
# @Function  :
import json
import traceback
from typing import Optional
from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status, Depends, Cookie, HTTPException
from fastapi.responses import StreamingResponse
from api.dlset_service import dlset_ask_websocket
from api.dlset_service_stream import dlset_ask_stream
from api.enum import ContentEnum
from api.schemas import DlsetQuestion
from nlq.core.graph import GraphWorkflow
from utils.public_utils import response_websocket
from utils.validate import validate_token, get_current_user

from utils.logging import getLogger
logger = getLogger()
router = APIRouter(prefix="/dlset", tags=["superset"])
load_dotenv()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, dlunifiedtoken: Optional[str] = Cookie(None)):
    try:
        get_current_user(dlunifiedtoken)
    except HTTPException as e:
        await websocket.close(code=1008, reason=e.detail)  # 关闭连接，代码1008表示违反策略
        return

    await websocket.accept()
    try:
        async for data in websocket.iter_text():
            question_json = json.loads(data)
            # question = DlsetQuestion(**question_json)
            # user_id = question.user_id
            # session_id = question.session_id
            user_id = question_json['user_id']
            session_id = question_json['session_id']
            try:
                jwt_token = question_json.get('token', None)
                answer = {}
                if jwt_token:
                    del question_json['token']
                    res = validate_token(jwt_token)
                    if not res['success']:
                        answer['X-Status-Code'] = status.HTTP_401_UNAUTHORIZED
                        await response_websocket(websocket=websocket, session_id=session_id, content=answer,
                                                 content_type=ContentEnum.END, user_id=user_id)
                    else:
                        # ask_result = await dlset_ask_websocket(websocket, question)
                        # logger.info(ask_result)
                        # answer = ask_result.dict()
                        # await response_websocket(websocket=websocket, session_id=session_id, content=answer,
                        #                          content_type=ContentEnum.END, user_id=user_id)
                        app = GraphWorkflow(graph_type="JSON")
                        async for info in app.astream_event_run(question_json):
                            answer_res = json.loads(info)
                            if "status" not in answer_res['content'].keys():
                                answer_res['X-Status-Code'] = 200
                                await websocket.send_text(json.dumps(answer_res, ensure_ascii=False))
                            else:
                                await websocket.send_text(info)
                        pass
                else:
                    answer['X-Status-Code'] = status.HTTP_401_UNAUTHORIZED
                    await response_websocket(websocket=websocket, session_id=session_id, content=answer,
                                             content_type=ContentEnum.END, user_id=user_id)
            except Exception as e:
                logger.error(e)
                msg = traceback.format_exc()
                logger.exception(msg)
                await response_websocket(websocket=websocket, session_id=session_id, content=msg,
                                         content_type=ContentEnum.EXCEPTION, user_id=user_id)
    except WebSocketDisconnect:
        logger.info(f"{websocket.client.host} disconnected.")


@router.post("/stream")
async def superset_stream(question: DlsetQuestion):
    question_json = question.dict()
    app = GraphWorkflow(graph_type="JSON")
    return StreamingResponse(app.astream_event_run(state=question_json, streamed_json=True), media_type="text/event-stream")