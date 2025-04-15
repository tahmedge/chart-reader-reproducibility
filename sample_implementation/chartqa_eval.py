import json
import pandas as pd

with open('t5_output.json', 'r') as file:
    data = json.load(file)

def relaxed_difference_match(output, prediction):
    try:
        output_num = float(output)
        prediction_num = float(prediction)
        # Handle zero explicitly
        if output_num == 0:
            return prediction_num == 0
        relaxed_diff = abs((output_num - prediction_num) / output_num)
        return relaxed_diff <= 0.05 
    except ValueError:
        return output.strip().lower() == prediction.strip().lower()

results_relaxed = []
for item in data:
    output = item['input_row']['output']
    prediction = item['prediction']
    match = relaxed_difference_match(output, prediction)
    results_relaxed.append({
        "question": item['input_row']['input'],
        "output": output,
        "prediction": prediction,
        "relaxed_match": match
    })

results_relaxed_df = pd.DataFrame(results_relaxed)

accuracy = results_relaxed_df['relaxed_match'].mean()
print(f"Accuracy based on relaxed match: {accuracy:.2%}")
