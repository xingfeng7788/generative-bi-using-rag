import io
import json
import traceback
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status, HTTPException, Cookie
from fastapi.responses import Response

from nlq.business.log_store import LogManagement
from nlq.business.profile import ProfileManagement
from nlq.core.graph import GraphWorkflow
from utils.logging import getLogger
from utils.public_utils import response_websocket
from utils.question import get_question_examples
from utils.validate import validate_token, get_current_user
from .enum import ContentEnum
from .schemas import Question, Answer, Option, CustomQuestion, FeedBackInput, HistoryRequest, Message, HistoryMessage, HistorySessionRequest
from . import service
from nlq.business.nlq_chain import NLQChain
from dotenv import load_dotenv

logger = getLogger()
router = APIRouter(prefix="/qa", tags=["qa"])
load_dotenv()


@router.get("/option", response_model=Option)
def option():
    return service.get_option()


@router.get("/get_custom_question", response_model=CustomQuestion)
def get_custom_question(data_profile: str):
    all_profiles = ProfileManagement.get_all_profiles_with_info()
    question_example = get_question_examples(all_profiles, data_profile)
    custom_question = CustomQuestion(custom_question=question_example)
    return custom_question


# @router.post("/ask", response_model=Answer)
# def ask(question: Question):
#     return service.ask(question)


@router.post("/get_history_by_user_profile")
def get_history_by_user_profile(history_request: HistoryRequest):
    user_id = history_request.user_id
    profile_name = history_request.profile_name
    log_type = history_request.log_type
    history_list = LogManagement.get_history(user_id, profile_name, log_type)
    chat_history = format_chat_history(history_list, log_type)
    return chat_history


def format_chat_history(history_list, log_type):
    chat_history = []
    chat_history_session = {}
    for item in history_list:
        session_id = item['session_id']
        query = item['query']
        if session_id not in chat_history_session:
            chat_history_session[session_id] = {
                "history": [],
                "title": query
            }
        log_info = item['log_info']
        human_message = Message(type="human", content=query)
        bot_message = Message(type="AI", content=json.loads(log_info))
        chat_history_session[session_id]['history'].append(human_message)
        chat_history_session[session_id]['history'].append(bot_message)
    for key, value in chat_history_session.items():
        each_session_history = HistoryMessage(session_id=key, messages=value['history'], title=value['title'])
        chat_history.append(each_session_history)
    return chat_history


@router.post("/user_feedback")
def user_feedback(input_data: FeedBackInput):
    feedback_type = input_data.feedback_type
    user_id = input_data.user_id
    session_id = input_data.session_id
    if feedback_type == "upvote":
        upvote_res = service.user_feedback_upvote(input_data.data_profiles, user_id, session_id, input_data.query,
                                                  input_data.query_intent, input_data.query_answer)
        return upvote_res
    else:
        downvote_res = service.user_feedback_downvote(input_data.data_profiles, user_id, session_id, input_data.query,
                                                      input_data.query_intent, input_data.query_answer)
        return downvote_res


@router.post("/get_sessions")
def get_sessions(history_request: HistoryRequest):
    return LogManagement.get_all_sessions(history_request.user_id, history_request.profile_name, history_request.log_type)


@router.get("/get_workflow_image")
async def get_workflow_image():
    app = GraphWorkflow()
    img = await app.get_graph_image()
    return Response(content=img, media_type="image/jpeg")


@router.post("/get_history_by_session")
def get_history_by_session(history_request: HistorySessionRequest):
    user_id = history_request.user_id
    history_list = LogManagement.get_all_history_by_session(profile_name=history_request.profile_name, user_id=user_id,
                                                            session_id=history_request.session_id,
                                                            size=1000, log_type=history_request.log_type)
    chat_history = format_chat_history(history_list, history_request.log_type)
    empty_history = {
        "session_id": history_request.session_id,
        "messages": [],
        "title": ""
    }
    if len(chat_history) > 0:
        return chat_history[0]
    else:
        return empty_history


@router.post("/delete_history_by_session")
def delete_history_by_session(history_request: HistorySessionRequest):
    user_id = history_request.user_id
    profile_name = history_request.profile_name
    session_id = history_request.session_id
    return LogManagement.delete_history_by_session(user_id, profile_name, session_id, log_type=history_request.log_type)


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
            question = Question(**question_json)
            session_id = question.session_id
            user_id = question.user_id
            try:
                jwt_token = question_json.get('dlunifiedtoken', None)
                if jwt_token:
                    del question_json['dlunifiedtoken']

                logger.info(f'---JWT TOKEN--- {jwt_token}')

                if jwt_token:
                    res = validate_token(jwt_token)
                    if not res["success"]:
                        answer = {}
                        answer['X-Status-Code'] = status.HTTP_401_UNAUTHORIZED
                        await response_websocket(websocket=websocket, session_id=session_id, content=answer,
                                                 content_type=ContentEnum.END, user_id=user_id)
                    else:
                        app = GraphWorkflow()
                        async for info in app.astream_event_run(question_json):
                            answer_res = json.loads(info)
                            if "status" not in answer_res['content'].keys():
                                answer_res['X-Status-Code'] = 200
                                answer_res['X-User-Id'] = res['user_id']
                                answer_res['X-User-Name'] = res['user_name']
                                await websocket.send_text(json.dumps(answer_res, ensure_ascii=False))
                            else:
                                await websocket.send_text(info)
                else:
                    answer = {}
                    answer['X-Status-Code'] = status.HTTP_401_UNAUTHORIZED
                    await response_websocket(websocket=websocket, session_id=session_id, content=answer,
                                             content_type=ContentEnum.END, user_id=user_id)
            except Exception:
                msg = traceback.format_exc()
                logger.exception(msg)
                await response_websocket(websocket=websocket, session_id=session_id, content=msg,
                                         content_type=ContentEnum.EXCEPTION, user_id=user_id)
    except WebSocketDisconnect:
        logger.info(f"{websocket.client.host} disconnected.")
