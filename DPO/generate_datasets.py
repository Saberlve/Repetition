import json
import argparse

def create_contrastive_dataset(icl_dataset_path, raw_dataset_path, output_path):
    """
    Read from icl_dataset.json and raw_dataset/minipile_train.json to create a contrastive dataset.

    Question: original_text from icl_dataset.json
    Reject: processed_result + icl_prompts (with original_text portion truncated)
    Chosen: matching entry from raw_dataset/minipile_train.json truncated to the same length as reject

    Args:
        icl_dataset_path: Path to the ICL dataset JSON file (icl_dataset.json)
        raw_dataset_path: Path to the raw dataset JSON file (minipile_train.json)
        output_path: Path to save the output JSON file
    """
    # Load the ICL dataset
    with open(icl_dataset_path, 'r', encoding='utf-8') as f:
        icl_data = json.load(f)

    # Load the raw dataset
    with open(raw_dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Create a dictionary for quick lookup of raw data entries by their first 50 characters
    raw_data_dict = {}
    for entry in raw_data:
        text = entry.get('text', '')
        if len(text) >= 50:
            # Use the first 50 characters as the key
            key = text[:50]
            raw_data_dict[key] = text

    # Create contrastive dataset
    contrastive_dataset = []
    for entry in icl_data:
        # Extract the question (original_text)
        question = entry.get('original_text', '')

        # Extract processed_result and icl_prompts
        processed_result = entry.get('processed_result', '')
        icl_prompts = entry.get('icl_prompts', '')

        # Create the reject by combining processed_result with icl_prompts
        # First, ensure we're not duplicating the original_text in the reject
        # by checking if processed_result starts with original_text
        if processed_result.startswith(question):
            # Remove the original_text portion from processed_result
            processed_result = processed_result[len(question):]

        # Combine processed_result with icl_prompts to create the reject
        reject = processed_result + icl_prompts

        # Find the matching entry in the raw dataset
        # Use the first 50 characters of the question as the key
        key = question[:50] if len(question) >= 50 else question

        if key in raw_data_dict:
            # Get the full text from the raw dataset
            full_text = raw_data_dict[key]

            # Truncate the full text to the same length as the reject
            chosen = full_text[len(question):]
            chosen = chosen[:len(reject)]

            # Add the entry to our dataset
            contrastive_dataset.append({    
                'question': question,
                'chosen': reject,   #reverse
                'reject': chosen,
            })

    # Save the contrastive dataset to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(contrastive_dataset, f, ensure_ascii=False, indent=4)

    print(f"Created {len(contrastive_dataset)} contrastive entries and saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create contrastive dataset from icl_dataset.json and raw_dataset/minipile_train.json')
    parser.add_argument('--icl_dataset_path', default='../dataset/icl_dataset.json',
                        help='Path to the ICL dataset JSON file (default: ../dataset/icl_dataset.json)')
    parser.add_argument('--raw_dataset_path', default='../raw_dataset/minipile_train.json',
                        help='Path to the raw dataset JSON file (default: ../raw_dataset/minipile_train.json)')
    parser.add_argument('--output_path', default='../dataset/contrastive_dataset.json',
                        help='Path to save the output JSON file (default: ../dataset/contrastive_dataset.json)')

    args = parser.parse_args()
    create_contrastive_dataset(args.icl_dataset_path, args.raw_dataset_path, args.output_path)
