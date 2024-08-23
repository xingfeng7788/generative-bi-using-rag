# Suggested Question
# PROFILE_QUESTION_TABLE_NAME = 'NlqSuggestedQuestion'
DEFAULT_PROMPT_NAME = 'suggested_question_prompt_default'
ACTIVE_PROMPT_NAME = 'suggested_question_prompt_active'
BEDROCK_MODEL_IDS = ['anthropic.claude-3-5-sonnet-20240620-v1:0', 'anthropic.claude-3-sonnet-20240229-v1:0',
                     'anthropic.claude-3-haiku-20240307-v1:0',
                     'mistral.mixtral-8x7b-instruct-v0:1', 'meta.llama3-70b-instruct-v1:0']
ENTITY_COMMENT_FORMAT = "{entity} is located in table {table_name}, column {column_name},  the dimension value is {value}."
en_cn_translation = {
    "intent_recognition": "意图识别",
    "reject_intent": "拒绝意图",
    "entity_retrieval": "实体检索",
    "qa_retrieval": "问答检索",
    "sql_generation": "SQL生成",
    "execute_query": "SQL执行",
    "analyze_data": "分析数据",
    "generate_sql_again": "重新生成SQL",
    "search_knowledge": "知识检索",
    "suggested_questions_generated": "生成建议问题",
    "answer_insights": "回答总结",
    "trigger_ask_in_reply": "触发反问反馈",
    "deal_ask_in_reply": "处理反问反馈",
    "end_format_answer": "输出结果",
    "deal_ask_dim_in_reply": "处理反问信息",
    "get_core_metadata": "获取场景元数据",
    "feedback_query_or_rewrite": "反馈问题或改写",
    "cot_generate_sub_task": "COT-生成子任务",
    "cot_entity_retrieval": "COT-实体检索",
    "cot_qa_retrieval": "COT-问答检索",
    "cot_sql_generation": "COT-SQL生成",
    "cot_sql_execute_query": "COT-SQL执行",
    "cot_data_visualization": "COT-数据可视化",
    "search_dataset_info": "获取数据集信息",
    "query_generation": "生成查询语句",
}