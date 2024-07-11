import json

path_project="/projects/0/prjs0871/hackathon/conv-search/DATA/"

# train set TOPIOCQA, index is row number
qrels = {}
with open(path_project+"raw_train.json", "r") as f:
    topiocqa_conv = json.load(f)
    with open(path_project+"queries_last.tsv", "w") as tsv_queries_last:
        with open(path_project+"queries_all.tsv", "w") as tsv_queries_all:
            for i, topic_turn in enumerate(topiocqa_conv):
                conv_turn_id = str(topic_turn["conv_id"])+"-"+str(topic_turn["turn_id"])
                user_question = topic_turn["question"].split("[SEP]")[-1].strip()

                all_user_question = topic_turn["question"].split("[SEP]")
                all_user_question = [t.strip() for t in all_user_question]
                all_user_question.reverse()
                tsv_queries_last.write(conv_turn_id+"\t"+user_question+"\n")
                tsv_queries_all.write(conv_turn_id+"\t"+" [SEP] ".join(all_user_question)+"\n")
                qrels[conv_turn_id] = {topic_turn["positive_ctxs"][0]["passage_id"]: 1}
json.dump(qrels, open(path_project+"qrel.json", "w"))

# test set TOPIOCQA, index is row number
with open(path_project+"raw_dev.json", "r") as f:
    topiocqa_conv = json.load(f)
    with open(path_project+"queries_rowid_dev_last.tsv", "w") as tsv_queries_last:
        with open(path_project+"queries_rowid_dev_all.tsv", "w") as tsv_queries_all:
            for i, topic_turn in enumerate(topiocqa_conv):
                conv_turn_id = str(topic_turn["conv_id"])+"-"+str(topic_turn["turn_id"])
                user_question = topic_turn["question"].split("[SEP]")[-1].strip()

                all_user_question = topic_turn["question"].split("[SEP]")
                all_user_question.reverse()
                tsv_queries_last.write(str(i)+"\t"+user_question+"\n")
                tsv_queries_all.write(str(i)+"\t"+" ".join(all_user_question)+"\n")
                qrels[str(i)] = {topic_turn["positive_ctxs"][0]["passage_id"]: 1}
        # mapturnid2rowid[turn_id]=str(i)
json.dump(qrels, open(path_project+"qrel_rowid_dev.json", "w"))