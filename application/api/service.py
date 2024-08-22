import base64
import json
import os
from typing import Union
from dotenv import load_dotenv
from nlq.business.connection import ConnectionManagement
from nlq.business.nlq_chain import NLQChain
from nlq.business.profile import ProfileManagement
from nlq.business.vector_store import VectorStore
from nlq.business.log_store import LogManagement
from utils.apis import get_sql_result_tool
from utils.database import get_db_url_dialect
from nlq.business.suggested_question import SuggestedQuestionManagement as sqm
from utils.domain import SearchTextSqlResult
from utils.llm import text_to_sql, get_query_intent, create_vector_embedding_with_sagemaker, \
    sagemaker_to_sql, sagemaker_to_explain, knowledge_search, get_agent_cot_task, data_analyse_tool, \
    generate_suggested_question, data_visualization, get_query_rewrite, optimize_query
from utils.opensearch import get_retrieve_opensearch
from utils.env_var import opensearch_info
from utils.public_utils import response_websocket
from utils.question import deal_comments
from utils.text_search import normal_text_search, agent_text_search
from utils.tool import generate_log_id, get_current_time, get_generated_sql_explain, get_generated_sql, \
    add_row_level_filter, change_class_to_str
from .schemas import Question, Answer, Example, Option, SQLSearchResult, AgentSearchResult, KnowledgeSearchResult, \
    TaskSQLSearchResult, ChartEntity, Message, AskReplayResult, HistoryMessage
from .exception_handler import BizException
from utils.constant import BEDROCK_MODEL_IDS, ACTIVE_PROMPT_NAME
from .enum import ErrorEnum, ContentEnum
from fastapi import WebSocket

from utils.logging import getLogger

logger = getLogger()

load_dotenv()


def get_option() -> Option:
    all_profiles = ProfileManagement.get_all_profiles_with_info()
    option = Option(
        data_profiles=all_profiles.keys(),
        bedrock_model_ids=BEDROCK_MODEL_IDS,
    )
    return option


def user_feedback_upvote(data_profiles: str, user_id: str, session_id: str, query: str, query_intent: str,
                         query_answer):
    try:
        if query_intent == "normal_search":
            VectorStore.add_sample(data_profiles, query, query_answer, "SQL")
        elif query_intent == "agent_search":
            VectorStore.add_sample(data_profiles, query, query_answer, "SQL")
            # VectorStore.add_agent_cot_sample(data_profiles, query, "\n".join(query_list))
        return True
    except Exception as e:
        return False


def user_feedback_downvote(data_profiles: str, user_id: str, session_id: str, query: str, query_intent: str,
                           query_answer):
    try:
        if query_intent == "normal_search":
            log_id = generate_log_id()
            current_time = get_current_time()
            LogManagement.add_log_to_database(log_id=log_id, user_id=user_id, session_id=session_id,
                                              profile_name=data_profiles,
                                              sql=query_answer, query=query,
                                              intent="normal_search_user_downvote",
                                              log_info="",
                                              time_str=current_time,
                                              log_type="feedback_downvote")
        elif query_intent == "agent_search":
            log_id = generate_log_id()
            current_time = get_current_time()
            LogManagement.add_log_to_database(log_id=log_id, user_id=user_id, session_id=session_id,
                                              profile_name=data_profiles,
                                              sql=query_answer, query=query,
                                              intent="agent_search_user_downvote",
                                              log_info="",
                                              time_str=current_time,
                                              log_type="feedback_downvote")
        return True
    except Exception as e:
        return False
