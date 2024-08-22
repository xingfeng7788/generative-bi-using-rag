def get_question_examples(all_profiles, data_profile):
    comments = all_profiles[data_profile]['comments']
    return deal_comments(comments)


def deal_comments(comments):
    comments_questions = []
    if len(comments.split("Examples:")) > 1:
        comments_questions_txt = comments.split("Examples:")[1]
        comments_questions = [i for i in comments_questions_txt.split("\n") if i != '']
    return comments_questions