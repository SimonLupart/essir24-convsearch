import argparse
import json

from splade.evaluation.utils.metrics import mrr_k, evaluate, judged_k

def load_and_evaluate(qrel_file_path, run_file_path, metric):
    with open(qrel_file_path) as reader:
        qrel = json.load(reader)
    with open(run_file_path) as reader:
        run = json.load(reader)

    if "TREC" in qrel_file_path:
        assert ("binary" not in qrel_file_path) == (metric == "ndcg" or metric == "ndcg_cut")
    if metric == "mrr_10":
        res = mrr_k(run, qrel, k=10)
        print("MRR@10:", res)
        return {"mrr_10": res}
    elif metric == "mrr_1000":
        res = mrr_k(run, qrel, k=1000)
        print("MRR@1000:", res)
        return {"mrr_1000": res}
    elif metric == "judged@10":
        res = judged_k(run, qrel, k=10)
        print("judged@10:", res)
        return {"judged@10": res}
    elif metric == "judged@100":
        res = judged_k(run, qrel, k=100)
        print("judged@100:", res)
        return {"judged@100": res}
    elif metric == "judged@1000":
        res = judged_k(run, qrel, k=1000)
        print("judged@1000:", res)
        return {"judged@1000": res}
    else:
        res = evaluate(run, qrel, metric=metric)
        print(metric, "==>", res)
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel_file_path")
    parser.add_argument("--run_file_path")
    parser.add_argument("--metric", default="mrr_10")
    args = parser.parse_args()
    load_and_evaluate(args.qrel_file_path, args.run_file_path, args.metric)
