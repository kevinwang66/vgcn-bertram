import json
import re

def add_spaces_around_key_in_values(json_file_path, output_file_path):
    # Load the JSON data from the input file
    with open(json_file_path, 'r') as infile:
        data = json.load(infile)
    
    # Iterate through each key in the dictionary
    for key, value_list in data.items():
        # Check if the key starts with '##', and strip '##' for matching
        stripped_key = key.lstrip('#')
        
        # Iterate through each URL in the list
        for i in range(len(value_list)):
            # Convert the value to lowercase
            modified_value = value_list[i].lower()
            
            # Add spaces around the key, after removing '##' for matching purposes
            modified_value = re.sub(r'(' + re.escape(stripped_key) + r')', r' \1 ', modified_value)
            
            # If the original key had '##', add it back in the correct place
            if key.startswith('##'):
                modified_value = modified_value.replace(stripped_key, '##' + stripped_key)
            
            # Replace '://', '-', ',', '.', '/', and double spaces with spaces
            modified_value = modified_value.replace('://', ' ')
            modified_value = modified_value.replace('-', ' ')
            modified_value = modified_value.replace(',', ' ')
            modified_value = modified_value.replace('.', ' ')
            modified_value = modified_value.replace('/', ' ')
            
            # Replace any double spaces with a single space
            modified_value = modified_value.replace('  ', ' ')
            
            # Update the value in the list
            value_list[i] = modified_value.strip()  # Strip any leading/trailing spaces
    
    # Save the modified data to the output file
    with open(output_file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

# Example usage
add_spaces_around_key_in_values('data/rare_words_with_contexts_test0.1_th=20_maxctx=5.json',
                                'data/processed_rare_words_with_contexts_test0.1_th=20_maxctx=5.json')

print("DONE!")