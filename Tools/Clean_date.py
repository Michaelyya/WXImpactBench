def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        csv_reader = csv.reader(infile)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            if len(row) >= 2:
                date, content = row[0], row[1]
                if content and content != '[]':
                    try:
                        # Parse the string as a Python list
                        content_list = ast.literal_eval(content)
                        # Join the list elements into a single string
                        text = ' '.join(content_list)
                        # Remove extra spaces and newlines
                        text = re.sub(r'\s+', ' ', text).strip()
                        # Write to the output file
                        outfile.write(f"{date}: {text}\n\n")
                    except (SyntaxError, ValueError):
                        print(f"Error processing row: {row}")

# Use the function
input_file = 'blog/blizzard_English_historical_ML_corpus.txt'
output_file = 'processed_output.txt'
process_file(input_file, output_file)
