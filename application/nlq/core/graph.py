# -*- coding: UTF-8 -*-
# @Project ：generative-bi-using-rag 
# @FileName ：graph.py
# @Author ：dingtianlu
# @Date ：2024/8/19 16:14
# @Function  :
import json
import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from nlq.business.connection import ConnectionManagement
from utils.apis import get_sql_result_tool
from utils.constant import ENTITY_COMMENT_FORMAT, en_cn_translation
from utils.opensearch import get_retrieve_opensearch
from utils.public_utils import response_stream_json, execute_sql
from api.enum import ErrorEnum, ContentEnum
from api.exception_handler import BizException
from api.schemas import KnowledgeSearchResult, SQLSearchResult, AgentSearchResult, AskReplayResult, Answer, ChartEntity, \
    TaskSQLSearchResult, GraphState, JSONSearchResult
from nlq.business.log_store import LogManagement
from nlq.business.profile import ProfileManagement
from utils.llm import get_query_intent, knowledge_search, get_query_rewrite, text_to_sql, optimize_query, \
    data_visualization, generate_suggested_question, get_agent_cot_task, data_analyse_tool, text_to_json
from utils.question import deal_comments
from utils.tool import change_class_to_str, generate_log_id, get_current_time, get_generated_sql, add_row_level_filter, \
    get_generated_sql_explain, get_generated_json, get_generated_think

from utils.logging import getLogger
logger = getLogger()

ask_result = AskReplayResult(query_rewrite="")

sql_search_result = SQLSearchResult(sql_data=[], sql="", data_show_type="table",
                                    sql_gen_process="",
                                    data_analyse="", sql_data_chart=[])

json_search_result = JSONSearchResult(
    json="-1",
    think_process="-1"
)

agent_search_response = AgentSearchResult(agent_summary="", agent_sql_search_result=[])
log_id = generate_log_id()
current_time = get_current_time()
knowledge_search_result = KnowledgeSearchResult(knowledge_response="")
MAX_EXECUTE_QUERY_NUMBER = 3


def handle_intent_recognition(state):
    # 调用bedrock模型进行意图识别
    intent_response = get_query_intent(state['bedrock_model_id'], state['query'],
                                       state['database_profile']['prompt_map'])
    intent = intent_response.get("intent", "normal_search")
    if intent == "agent_search":
        if not state['agent_cot_flag']:
            if state['graph_type'] == "SQL":
                intent = "normal_search"
            elif state['graph_type'] == "JSON":
                intent = "reject_search"
    state['query_intent'] = intent
    state['entity_slots'] = intent_response.get("slot", [])
    return state


def handle_knowledge_search(state):
    database_profile = state['database_profile']
    question_example = deal_comments(database_profile['comments'])
    response = knowledge_search(search_box=state['query'],
                                model_id=state['bedrock_model_id'],
                                prompt_map=database_profile['prompt_map'],
                                dialect=database_profile['db_type'],
                                table_info=database_profile['tables_info'],
                                question_example=question_example)

    knowledge_search_result.knowledge_response = response
    answer = Answer(query=state['query'], query_rewrite=state['query_rewrite'],
                    query_intent="knowledge_search",
                    knowledge_search_result=knowledge_search_result,
                    sql_search_result=sql_search_result, agent_search_result=agent_search_response,
                    suggested_question=state['suggested_question_list'], ask_rewrite_result=ask_result,
                    json_search_result=json_search_result)
    state['answer'] = answer
    return state


def handle_reject_intent(state):
    answer = Answer(query=state['query'],
                    query_intent="reject_search",
                    knowledge_search_result=knowledge_search_result,
                    sql_search_result=sql_search_result,
                    agent_search_result=agent_search_response,
                    suggested_question=[],
                    ask_rewrite_result=ask_result,
                    json_search_result=json_search_result
                    )
    state['answer'] = answer
    return state


def handle_agent_retrieval(state):
    agent_cot_retrieve = get_retrieve_opensearch(state['query_rewrite'], "agent", state['profile_name'], 2, 0.5)
    state['agent_cot_retrieves'] = agent_cot_retrieve
    return state


def handle_generate_cot_sub_task(state):
    database_profile = state['database_profile']
    agent_cot_task_result = get_agent_cot_task(state['bedrock_model_id'], database_profile['prompt_map'], state['query_rewrite'],
                                               database_profile['tables_info'],
                                               state['agent_cot_retrieves'])
    state['agent_cot_task'] = agent_cot_task_result
    return state


def handle_cot_entity_retrieval(state):
    for task, task_info in state['agent_cot_task'].items():
        entity_retrieve = get_retrieve_opensearch(task_info, "ner", state['profile_name'], 3, 0.5)
        if task_info not in state['cot_execute_query_info']:
            state['cot_execute_query_info'][task_info] = {}
        state['cot_execute_query_info'][task_info]["entity_retrieves"] = entity_retrieve
    return state


def handle_cot_qa_retrieval(state):
    for task, task_info in state['agent_cot_task'].items():
        qa_retrieve_result = get_retrieve_opensearch(task_info, "query", state['profile_name'], 3, 0.5)
        if task_info not in state['cot_execute_query_info']:
            state['cot_execute_query_info'][task_info] = {}
        state['cot_execute_query_info'][task_info]["qa_retrieves"] = qa_retrieve_result
    return state


def handle_cot_sql_generation(state):
    database_profile = state['database_profile']
    for task, task_info in state['agent_cot_task'].items():
        each_task_response = text_to_sql(database_profile['tables_info'],
                                         database_profile['hints'],
                                         database_profile['prompt_map'],
                                         task_info,
                                         model_id=state['bedrock_model_id'],
                                         sql_examples=state['cot_execute_query_info'][task_info]["qa_retrieves"],
                                         ner_example=state['cot_execute_query_info'][task_info]["entity_retrieves"],
                                         dialect=database_profile['db_type'],
                                         model_provider=None)
        each_task_sql = get_generated_sql(each_task_response)
        # 行级权限
        each_task_sql = add_row_level_filter(each_task_sql, database_profile['tables_info'])
        if os.getenv("SQL_OPTIMIZATION_ENABLED") == '1':
            optimized_response = optimize_query(database_profile['prompt_map'], state['bedrock_model_id'],
                                                each_task_sql,
                                                database_profile['db_type'])
            each_task_sql = get_generated_sql(optimized_response)
        if task_info not in state['cot_execute_query_info']:
            state['cot_execute_query_info'][task_info] = {}
        state['cot_execute_query_info'][task_info]["sql"] = each_task_sql
        state['cot_execute_query_info'][task_info]["sql_explain"] = get_generated_sql_explain(each_task_response)
    return state


def handle_cot_sql_execute_query(state):
    database_profile = state['database_profile']
    filter_deep_dive_sql_result = []
    agent_sql_search_result = []
    sql_list = []
    for task, task_info in state['agent_cot_task'].items():
        sql = state['cot_execute_query_info'][task_info]["sql"]
        each_task_res = get_sql_result_tool(database_profile, sql)
        if each_task_res["status_code"] == 200 and len(each_task_res["data"]) > 0:
            filter_deep_dive_sql_result.append(
                {
                    "sql": sql,
                    "data_result": each_task_res["data"].to_json(orient='records'),
                    "sql_explain": state['cot_execute_query_info'][task_info]["sql_explain"]
                }
            )
            model_select_type, show_select_data, select_chart_type, show_chart_data = data_visualization(
                state["bedrock_model_id"],
                task_info,
                each_task_res["data"],
                database_profile['prompt_map'])

            sub_task_sql_result = SQLSearchResult(sql_data=show_select_data, sql=sql,
                                                  data_show_type=model_select_type,
                                                  sql_gen_process=state['cot_execute_query_info'][task_info][
                                                      "sql_explain"],
                                                  data_analyse="", sql_data_chart=[])
            if select_chart_type != "-1":
                sub_sql_chart_data = ChartEntity(chart_type="", chart_data=[])
                sub_sql_chart_data.chart_type = select_chart_type
                sub_sql_chart_data.chart_data = show_chart_data
                sub_task_sql_result.sql_data_chart = [sub_sql_chart_data]

            each_task_sql_search_result = TaskSQLSearchResult(sub_task_query=task_info,
                                                              sql_search_result=sub_task_sql_result)
            agent_sql_search_result.append(each_task_sql_search_result)
            sql_list.append(sql)

        else:
            logger.warning(task_info + "The SQL error Info: ")
    state["filter_deep_dive_sql_result"] = filter_deep_dive_sql_result
    state["agent_sql_search_result"] = agent_sql_search_result
    state['sql'] = json.dumps(sql_list, ensure_ascii=False)
    return state


def handle_cot_data_visualization(state):
    agent_data_analyse_result = data_analyse_tool(state['bedrock_model_id'], state['database_profile']['prompt_map'],
                                                  state['query'],
                                                  json.dumps(state['filter_deep_dive_sql_result'], ensure_ascii=False),
                                                  "agent")
    logger.info("agent_data_analyse_result")
    logger.info(agent_data_analyse_result)
    agent_search_response.agent_summary = agent_data_analyse_result
    agent_search_response.agent_sql_search_result = state['agent_sql_search_result']

    answer = Answer(query=state['query'], query_intent="agent_search", knowledge_search_result=knowledge_search_result,
                    sql_search_result=sql_search_result, agent_search_result=agent_search_response,
                    suggested_question=state['suggested_question_list'], ask_rewrite_result=ask_result
                    , json_search_result=json_search_result)
    agent_answer_info = change_class_to_str(answer)
    LogManagement.add_log_to_database(log_id=log_id, user_id=state['user_id'], session_id=state['session_id'],
                                      profile_name=state['profile_name'], sql="",
                                      query=state['query'],
                                      intent="agent_search",
                                      log_info=agent_answer_info,
                                      log_type=state['graph_type'],
                                      time_str=current_time)
    state['answer'] = answer
    return state


def handle_entity_retrieval(state):
    entity_slot_retrieves = []
    entity_name_set = set()
    for each_entity in state['entity_slots']:
        entity_retrieve = get_retrieve_opensearch(each_entity, "ner", state['profile_name'], 1, 0.7)
        if len(entity_retrieve) > 0:
            for each_entity_retrieve in entity_retrieve:
                if each_entity_retrieve['_source']['entity'] not in entity_name_set:
                    entity_name_set.add(each_entity_retrieve['_source']['entity'])
                    entity_slot_retrieves.append(each_entity_retrieve)
    state['entity_slot_retrieves'] = entity_slot_retrieves
    same_name_entity = {}
    for each_entity in entity_slot_retrieves:
        if each_entity['_source']['entity_count'] > 1 and each_entity['_score'] > 0.98:
            same_name_entity[each_entity['_source']['entity']] = each_entity['_source']['entity_table_info']
    if len(same_name_entity) > 0:
        # 如果正常流程就会触发反问，反问
        if state['context_state'] != "ask_dim_in_reply":
            state['context_state'] = "ask_dim_in_reply"
            state['ask_entity_select'] = same_name_entity
            answer = Answer(query=state['query'], query_intent="normal_search",
                            knowledge_search_result=knowledge_search_result,
                            sql_search_result=sql_search_result, agent_search_result=agent_search_response,
                            suggested_question=state['suggested_question_list'], ask_rewrite_result=ask_result
                            , json_search_result=json_search_result)
            state['answer'] = answer
    return state


def handle_qa_retrieval(state):
    retrieve_results = []
    if state['use_rag_flag']:
        retrieve_results = get_retrieve_opensearch(state['query'], "query", state['profile_name'], 3, 0.5,
                                                   sample_type=state['graph_type'])
    state['qa_retrieves'] = retrieve_results
    return state


def handle_sql_generation(state):
    database_profile = state['database_profile']
    response = text_to_sql(database_profile['tables_info'],
                           database_profile['hints'],
                           database_profile['prompt_map'],
                           state['query'],
                           model_id=state['bedrock_model_id'],
                           sql_examples=state['qa_retrieves'],
                           ner_example=state['entity_slot_retrieves'],
                           dialect=database_profile['db_type'],
                           additional_info=state['additional_info'],
                           )
    logger.info(f'{response=}')
    sql = get_generated_sql(response)
    # 行级权限
    sql = add_row_level_filter(sql, database_profile['tables_info'])
    if os.getenv("SQL_OPTIMIZATION_ENABLED") == '1':
        optimized_response = optimize_query(database_profile['prompt_map'], state['bedrock_model_id'], sql,
                                            database_profile['db_type'])
        sql = get_generated_sql(optimized_response)
    if state['explain_gen_process_flag']:
        state['sql_gen_process'] = get_generated_sql_explain(response)
    state['sql'] = sql
    return state


def handle_search_dataset_info(state):
    # 读取superset 数据集相关信息
    dataset_info_sql = f"""
    select column_name 字段,
     case when groupby=1 then '维度' else '度量' end as 维度度量,
     verbose_name verbose_name, filterable 是否可用于过滤, is_dttm 是否为时间列 from table_columns where table_id={state['table_id']}"""

    dataset_info = execute_sql(dataset_info_sql)
    # 处理数据集信息 按照csv的格式处理
    dataset_info_csv = ["字段，维度度量，verbose_name，是否可用于过滤，是否为时间列"]
    for each_info in dataset_info:
        dataset_info_csv.append(
            f"        {each_info['字段']},{each_info['维度度量']},{each_info['verbose_name']},{each_info['是否可用于过滤']},{each_info['是否为时间列']}")
    dataset_info_csv = "\n".join(dataset_info_csv)

    table_info_sql = f"""
    select `schema`, table_name, description, main_dttm_col from tables where id={state['table_id']}
    """
    table_info = execute_sql(table_info_sql)[0]

    # 读取数据集数据
    dataset_schema = f"""
    dataset_id: {state['table_id']}
    dataset_name: {table_info['schema']}.{table_info['table_name']}
    dataset_description: {table_info['description']}
    主时间列: {table_info['main_dttm_col']}
    数据集列信息: 
    {dataset_info_csv}
    """
    state['dataset_schema'] = dataset_schema
    return state


def handle_execute_json_generation(state):
    database_profile = state['database_profile']
    response = text_to_json(
        state['dataset_schema'],
        database_profile['tables_info'],
        database_profile['hints'],
        database_profile['prompt_map'],
        state['query'],
        model_id=state['bedrock_model_id'],
        sql_examples=state['qa_retrieves'],
        ner_example=state['entity_slot_retrieves'],
        dialect=database_profile['db_type'],
        model_provider=None)
    logger.info(f'{response=}')
    json_str = get_generated_json(response)
    think_process = get_generated_think(response)
    answer = Answer(query=state['query'], query_intent="normal_search", knowledge_search_result=knowledge_search_result,
                    sql_search_result=sql_search_result, agent_search_result=agent_search_response,
                    suggested_question=state['suggested_question_list'], ask_rewrite_result=ask_result
                    , json_search_result=JSONSearchResult(json=json_str, think_process=think_process))
    state['sql'] = json_str
    state['answer'] = answer
    return state


def handle_answer_with_insights(state):
    if state['answer_with_insights']:
        insight_result = data_analyse_tool(state['bedrock_model_id'],
                                           state['database_profile']['prompt_map'],
                                           state['query_rewrite'],
                                           state['execute_query_info'][-1]['data'].to_json(orient='records', force_ascii=False),
                                           "query")
        state['insight_result'] = insight_result
    return state


def handle_execute_query(state):
    results = get_sql_result_tool(state['database_profile'], state['sql'])
    if results.get("status_code") == 500:
        execute_query_info = {"success": False, "error_info": results.get("error_info"), "data": None,
                              "sql": state['sql']}
    else:
        execute_query_info = {"success": True, "data": results.get("data"), "sql": state['sql']}
    state['execute_query_info'].append(execute_query_info)
    return state


def handle_analyze_data(state):
    model_select_type, show_select_data, select_chart_type, show_chart_data = data_visualization(
        state['bedrock_model_id'],
        state['query_rewrite'],
        state['execute_query_info'][-1]['data'],
        state['database_profile']['prompt_map'])
    if select_chart_type != "-1":
        sql_chart_data = ChartEntity(chart_type="", chart_data=[])
        sql_chart_data.chart_type = select_chart_type
        sql_chart_data.chart_data = show_chart_data
        sql_search_result.sql_data_chart = [sql_chart_data]
    sql_search_result.sql_gen_process = state["sql_gen_process"]
    sql_search_result.sql = state['sql']
    sql_search_result.sql_data = show_select_data
    sql_search_result.data_show_type = model_select_type
    sql_search_result.data_analyse = state['insight_result']

    answer = Answer(query=state['query'], query_rewrite=state['query_rewrite'], query_intent="normal_search",
                    knowledge_search_result=knowledge_search_result,
                    sql_search_result=sql_search_result, agent_search_result=agent_search_response,
                    suggested_question=state['suggested_question_list'], ask_rewrite_result=ask_result,
                    json_search_result=json_search_result
                    )
    state['answer'] = answer
    return state


def handle_generate_sql_again(state):
    error_info = []
    for exec_info in state['execute_query_info']:
        error_info.append(
            '''\n NOTE: when I try to write a SQL <sql>{sql_statement}</sql>, I got an error <error>{error}</error>. Please consider and avoid this problem. '''.format(
                sql_statement=exec_info['sql'],
                error=exec_info["error_info"]))
    additional_info = "\n".join(error_info)
    state['additional_info'] = additional_info
    return state


def handle_ask_dim_in_reply(state):
    # 维度信息反问
    update_entity_slot_retrieves = []
    for each_entity in state['entity_slot_retrieves']:
        if each_entity['_source']['entity'] in state['entity_selected']:
            selected_info = state['ask_entity_select'][each_entity['_source']['entity']]
            each_entity['_source']['comment'] = ENTITY_COMMENT_FORMAT.format(entity=each_entity['_source']['entity'],
                                                                             table_name=selected_info['table_name'],
                                                                             column_name=selected_info['column_name'],
                                                                             value=selected_info['value'])
        update_entity_slot_retrieves.append(each_entity)
    state['entity_slot_retrieves'] = update_entity_slot_retrieves
    return state


def handle_feedback_query_or_rewrite(state):
    query_rewrite_result = {"intent": "original_problem", "query": state['query']}
    if state['context_window'] > 0:
        user_query_history = LogManagement.get_history_by_session(profile_name=state['profile_name'],
                                                                  user_id=state['user_id'],
                                                                  session_id=state['session_id'],
                                                                  size=state['context_window'],
                                                                  log_type='chat_history')
        if len(user_query_history) > 0:
            user_query_history.append("user:" + state['query'])
            logger.info("The Chat history is {history}".format(history="\n".join(user_query_history)))
            query_rewrite_result = get_query_rewrite(state['bedrock_model_id'], state['query'],
                                                     state['database_profile']['prompt_map'], user_query_history)
            logger.info(
                "The query_rewrite_result is {query_rewrite_result}".format(query_rewrite_result=query_rewrite_result))
    state['query_rewrite'] = query_rewrite_result.get("query")
    state['query_rewrite_intent'] = query_rewrite_result.get("intent")
    return state


def get_profile_info(state):
    selected_profile = state["profile_name"]
    all_profiles = ProfileManagement.get_all_profiles_with_info()
    if selected_profile not in all_profiles:
        raise BizException(ErrorEnum.PROFILE_NOT_FOUND)
    database_profile = all_profiles[selected_profile]
    if database_profile['db_url'] == '':
        conn_name = database_profile['conn_name']
        db_url = ConnectionManagement.get_db_url_by_name(conn_name)
        database_profile['db_url'] = db_url
        database_profile['db_type'] = ConnectionManagement.get_db_type_by_name(conn_name)
    state['database_profile'] = database_profile
    return state


def handle_ask_in_reply(state):
    answer = Answer(query=state['query'], query_rewrite=state['query_rewrite'], query_intent="ask_in_reply",
                    knowledge_search_result=knowledge_search_result,
                    sql_search_result=sql_search_result, agent_search_result=agent_search_response,
                    suggested_question=[], ask_rewrite_result=ask_result, json_search_result=json_search_result)
    state['answer'] = answer
    return state


def handle_suggested_question_list(state):
    if state['gen_suggested_question_flag']:
        generated_sq = generate_suggested_question(state['database_profile']['prompt_map'], state['query_rewrite'],
                                                   state['bedrock_model_id'])
        split_strings = generated_sq.split("[generate]")
        generate_suggested_question_list = [s.strip() for s in split_strings if s.strip()]
        state['suggested_question_list'] = generate_suggested_question_list
    return state


def decide_choose_next_node(state):
    # initial 正常流程, ask_in_reply: 缺少时间粒度反问, ask_dim_in_reply: 缺少维度信息反问
    if state['context_state'] == 'initial':
        return "normal_process"
    elif state['context_state'] == 'ask_in_reply':
        return "ask_in_reply"
    elif state['context_state'] == 'ask_dim_in_reply':
        return "ask_dim_in_reply"
    else:
        raise Exception("Invalid state")


def decide_is_need_regenerate_sql(state):
    if not state['execute_query_info'][-1]['success'] and len(state['execute_query_info']) < MAX_EXECUTE_QUERY_NUMBER:
        return "regenerate_sql"
    else:
        return "not_regenerate_sql"


def decide_intent_recognition_next_node(state):
    return state.get('query_intent')


def decide_reject_next_node(state):
    if state.get('query_intent') == "reject_search":
        return "reject_search"
    else:
        return "not_reject_search"


def decide_query_rewrite_next_node(state):
    if state.get('query_rewrite_intent') in ['original_problem', 'rewrite_question']:
        return "not_ask_in_reply"
    else:
        return "ask_in_reply"


def decide_sql_json_next_node(state):
    if state['graph_type'] == 'SQL':
        return "sql"
    else:
        return "json"


def handle_output_format_answer(state):
    state['answer'].context_state = state['context_state']
    state['answer'].ask_entity_select = state['ask_entity_select']
    state['answer'].entity_slots = state['entity_slots']
    state['answer'].entity_slot_retrieves = state['entity_slot_retrieves']
    state['answer'].qa_retrieves = state['qa_retrieves']
    state['answer'].agent_cot_retrieves = state['agent_cot_retrieves']
    state['answer'].agent_cot_task = state['agent_cot_task']
    if state['context_state'] != 'ask_dim_in_reply':
        answer_info = change_class_to_str(state['answer'])
        LogManagement.add_log_to_database(log_id=log_id, user_id=state['user_id'], session_id=state['session_id'],
                                          profile_name=state['profile_name'], sql=state["sql"], query=state['query'],
                                          intent=state['query_intent'],
                                          log_info=answer_info,
                                          log_type=state['graph_type'],
                                          time_str=current_time)
    return state


class GraphWorkflow:
    def __init__(self, graph_type: str = "SQL"):
        self.type = graph_type
        self.workflow = StateGraph(GraphState)
        self.build_workflow()
        self.app = self.workflow.compile()

    def add_nodes(self):
        nodes = [
            ("intent_recognition", handle_intent_recognition),
            ("reject_intent", handle_reject_intent),
            ("entity_retrieval", handle_entity_retrieval),
            ("qa_retrieval", handle_qa_retrieval),
            ("sql_generation", handle_sql_generation),
            ("execute_query", handle_execute_query),
            # ("agent_task", handle_agent_task),
            ("analyze_data", handle_analyze_data),
            ("generate_sql_again", handle_generate_sql_again),
            ("search_knowledge", handle_knowledge_search),
            ("suggested_questions_generated", handle_suggested_question_list),
            ("answer_insights", handle_answer_with_insights),
            ("trigger_ask_in_reply", handle_ask_in_reply),
            # ("deal_ask_in_reply", deal_ask_in_reply),
            ("end_format_answer", handle_output_format_answer),
            ("deal_ask_dim_in_reply", handle_ask_dim_in_reply),
            ("get_core_metadata", get_profile_info),
            ("feedback_query_or_rewrite", handle_feedback_query_or_rewrite),
            # cot
            ("agent_retrieval", handle_agent_retrieval),
            ("cot_generate_sub_task", handle_generate_cot_sub_task),
            ("cot_entity_retrieval", handle_cot_entity_retrieval),
            ("cot_qa_retrieval", handle_cot_qa_retrieval),
            ("cot_sql_generation", handle_cot_sql_generation),
            ("cot_sql_execute_query", handle_cot_sql_execute_query),
            ("cot_data_visualization", handle_cot_data_visualization),
            # superset
            ("search_dataset_info", handle_search_dataset_info),
            ("query_generation", handle_execute_json_generation),
        ]

        for node_name, handler in nodes:
            self.workflow.add_node(node_name, handler)

    def add_edges(self):
        edges = [
            ('agent_retrieval', "cot_generate_sub_task"),
            ('cot_generate_sub_task', "cot_entity_retrieval"),
            ('cot_entity_retrieval', "cot_qa_retrieval"),
            ('cot_qa_retrieval', "cot_sql_generation"),
            ('cot_sql_generation', "cot_sql_execute_query"),
            ('cot_sql_execute_query', "cot_data_visualization"),
            ('cot_data_visualization', "end_format_answer"),
            ('reject_intent', "end_format_answer"),
            ('trigger_ask_in_reply', "end_format_answer"),
            ('search_knowledge', "end_format_answer"),
            # ('agent_task', "end_format_answer"),
            ('analyze_data', "end_format_answer"),
            ("deal_ask_dim_in_reply", "qa_retrieval"),
            ('search_dataset_info', 'query_generation'),
            ('query_generation', 'end_format_answer'),
            ("sql_generation", "execute_query"),
            ("answer_insights", "analyze_data"),
            ("generate_sql_again", "sql_generation"),
        ]

        for source, target in edges:
            self.workflow.add_edge(source, target)

    def build_workflow(self):
        self.add_nodes()
        self.add_edges()
        self.add_conditional_edges()
        self.workflow.set_entry_point("get_core_metadata")
        self.workflow.add_edge("end_format_answer", END)

    def add_conditional_edges(self):
        conditional_edges = [
            ("get_core_metadata", decide_choose_next_node, {
                "normal_process": "feedback_query_or_rewrite",
                # "ask_in_reply": "deal_ask_in_reply",
                "ask_dim_in_reply": "deal_ask_dim_in_reply",
            }),
            ("feedback_query_or_rewrite", decide_query_rewrite_next_node, {
                "not_ask_in_reply": "intent_recognition",
                "ask_in_reply": "trigger_ask_in_reply",
            }),
            ("intent_recognition", decide_reject_next_node, {
                "reject_search": "reject_intent",
                "not_reject_search": "suggested_questions_generated"
            }),
            ("suggested_questions_generated", decide_intent_recognition_next_node, {
                "agent_search": "agent_retrieval",
                "knowledge_search": "search_knowledge",
                "normal_search": "entity_retrieval",
            }),
            ("entity_retrieval", decide_choose_next_node, {
                "normal_process": "qa_retrieval",
                "ask_dim_in_reply": "end_format_answer",
            }),
            ("qa_retrieval", decide_sql_json_next_node, {
                "sql": "sql_generation",
                "json": "search_dataset_info",
            }),
            ("execute_query", decide_is_need_regenerate_sql, {
                "regenerate_sql": "generate_sql_again",
                "not_regenerate_sql": "answer_insights",
            })
        ]

        for node, decision_func, branches in conditional_edges:
            self.workflow.add_conditional_edges(node, decision_func, branches)

    async def run(self, state):
        state = self.init_state(state)
        return self.app.invoke(state)

    async def astream_event_run(self, state, streamed_json=False):
        state = self.init_state(state)
        async for event in self.app.astream_events(state, config={"configurable": {"thread_id": 1}}, version="v2"):
            # 1. 记录on_chain_start、on_chain_end 事件
            # 2. "LangGraph", "__start__" ChannelWrite* 事件不记录
            # 3. decide_ 开头的条件事件不记录
            # 4. end_format_answer 最终节点输出结果
            if ((event["event"] in ["on_chain_start", "on_chain_end"]
                 and event['name'] not in ["LangGraph", "__start__"]
                 and not event['name'].startswith("ChannelWrite"))):
                if event["event"] == "on_chain_end" and event['name'] == "end_format_answer":
                    async for data in response_stream_json(state['session_id'],
                                                           event['data']['output']['answer'].dict(),
                                                           ContentEnum.END,
                                                           "-1",
                                                           state['user_id'], streamed_json):
                        yield data
                        break
                else:
                    if not event['name'].startswith("decide_"):
                        async for data in response_stream_json(state['session_id'], en_cn_translation.get(event['name'], event['name']), ContentEnum.STATE,
                                                               event["event"].split("_")[-1],
                                                               state['user_id'], streamed_json):
                            yield data

    async def get_graph_image(self):
        try:
            return self.app.get_graph().draw_mermaid_png()
        except Exception:
            pass

    def init_state(self, state: Dict[str, Any]) -> GraphState:
        default_state = self.default_graph_state()
        for key, value in default_state.items():
            if key not in state:
                state[key] = value
        return state

    def default_graph_state(self) -> GraphState:
        if self.type == "JSON":
            agent_cot_flag = False
        else:
            agent_cot_flag = True
        return {
            "query": "",
            "bedrock_model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "use_rag_flag": True,
            "visualize_results_flag": True,
            "intent_ner_recognition_flag": True,
            "agent_cot_flag": agent_cot_flag,
            "profile_name": "",
            "explain_gen_process_flag": True,
            "gen_suggested_question_flag": True,
            "answer_with_insights": False,
            "top_k": 250.0,
            "top_p": 0.9,
            "max_tokens": 2048,
            "temperature": 0.01,
            "context_window": 5,
            "session_id": "-1",
            "user_id": "admin",
            "database_profile": {},
            "query_rewrite_intent": "",
            "entity_slots": [],
            "entity_slot_retrieves": [],
            "qa_retrieves": [],
            "query_rewrite": "",
            "query_intent": "",
            "sql": "-1",
            "answer": None,
            "context_state": "initial",
            "execute_query_info": [],
            "additional_info": "",
            "suggested_question_list": [],
            "insight_result": "",
            "sql_gen_process": None,
            "agent_cot_retrieves": [],
            "agent_cot_task": {},
            "cot_execute_query_info": {},
            "filter_deep_dive_sql_result": [],
            "agent_sql_search_result": [],
            "is_debug": False,
            "table_id": None,
            "graph_type": self.type,
            "dataset_schema": "",
            "ask_entity_select": {},
            "entity_selected": {},
        }
