import argparse
import glob
import json
from os.path import join, basename, dirname

from prettytable import PrettyTable, HRuleStyle
from kina_bench.config import PROJECT_ROOT


def list_to_table(data_list, headers=None):
    table = PrettyTable()
    table.hrules = HRuleStyle.ALL

    max_list_length = max(len(value) for value in data_list)

    if headers is None:
        headers = [f"Value{i + 1}" for i in range(max_list_length)]

    table.field_names = headers

    for value_list in data_list:
        row = value_list + [""] * (max_list_length - len(value_list))
        table.add_row(row)

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        default="KINA-899",
        help="Result basename (without .json) under results/.../.../",
    )
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()

    result_dir = join(PROJECT_ROOT, "results")
    if args.backup:
        result_dir += "_backup"

    result_json_list = glob.glob(join(result_dir, f"*/*/{args.data_name}.json"))
    result_json_list.sort()

    print_list = []
    for result_json_path in result_json_list:
        if result_json_path.endswith("_bad_cases.json"):
            continue

        print(result_json_path)

        data_name = basename(result_json_path).split(".")[0]
        gen_name = basename(dirname(result_json_path))
        model_name = basename(dirname(dirname(result_json_path)))

        with open(result_json_path, "r") as f:
            result = json.load(f)

        score_list = []
        empty_pred_num = 0
        for doc in result:
            # extracted_predictions format: [[preds_for_resp1], [preds_for_resp2], ...]
            # e.g. when n_sampling=4: [['A', 'A'], ['B', 'B'], ['A', 'A'], ['C', 'C']]
            extracted_preds = doc["extracted_predictions"]

            # Verify: score > 0 if and only if gt appears in any response's extraction results
            gt_in_preds = any(doc["gt"] in preds for preds in extracted_preds)
            if doc["score"] > 0:
                assert gt_in_preds, (
                    f"Score > 0 but gt not in predictions: "
                    f"score={doc['score']}, gt={doc['gt']}, preds={extracted_preds}, id={doc['id']}"
                )
            else:
                assert not gt_in_preds, (
                    f"Score = 0 but gt in predictions: "
                    f"score={doc['score']}, gt={doc['gt']}, preds={extracted_preds}, id={doc['id']}"
                )

            score_list.append(doc["score"])

            # Count samples where no answer could be extracted from any response
            if all(len(preds) == 0 for preds in extracted_preds):
                empty_pred_num += 1

        if len(score_list) == 0:
            print(f"empty {result_json_path}")
            continue

        avg_score = sum(score_list) / len(score_list)
        print_list.append([data_name, model_name, gen_name, f"{avg_score:.2%}", len(score_list), empty_pred_num])

    print_list.sort(key=lambda x: x[1])

    def get_model_group(model_name):
        if '--' in model_name:
            return model_name.split('--', 1)[1].split('-', 1)[0]
        return model_name

    custom_headers = ["data", "model", "generation", "avg_score", "total_num", "empty_pred_num"]
    table = PrettyTable()
    table.field_names = custom_headers
    for i, row in enumerate(print_list):
        is_last = i == len(print_list) - 1
        diff_group = not is_last and get_model_group(row[1]) != get_model_group(print_list[i + 1][1])
        table.add_row(row, divider=(is_last or diff_group))
    print_table = table
    print(print_table)

    with open(f"{result_dir}/results.txt", "w") as f:
        f.write(str(print_table))


if __name__ == '__main__':
    main()
