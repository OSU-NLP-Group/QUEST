import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Extract prediction fields from JSON files')
    parser.add_argument('--input_dir', type=str, default="./outputs/openended_trajectories/", help='Input directory containing JSON files')
    parser.add_argument('--output_file', type=str, default='./outputs/extracted_questions.jsonl', help='Output JSONL file path')
    args = parser.parse_args()

    filelist = os.listdir(args.input_dir)
    filelist.sort(key=lambda x: int(x.split('_')[1]))
    out = []
    for idx, file in enumerate(filelist):
        with open(os.path.join(args.input_dir, file), "r") as f:
            data = json.load(f)
            out.append({
                'id': idx + 1,
                'topic': "_".join(file.split('_')[7:]).strip("json").strip('.'),
                "conceptual_breadth": data["prediction"]['json']['conceptual_breadth'],
                "logical_nesting": data["prediction"]['json']['logical_nesting'],
                "exploration": data["prediction"]['json']['exploration'],
                'prompt': data["prediction"]['json']['proposed_question']
            })
    print(len(out))
    with open(args.output_file, "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
