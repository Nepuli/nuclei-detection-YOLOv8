import os
import cv2
import pandas as pd
from ultralytics import YOLO

# List of your input folders
input_folders = [
    # Specify file paths of the folders here
]

OUTPUT_BASE_DIR = #TODO: # Output path for saving the results

model_path = #TODO: # File path of the pre-trained model

# Load the model
model = YOLO(model_path)  # Load your custom model

threshold = 0.3

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# List for the number of detected nuclei per image
all_count_nuclei_list = []

# Iterate over all input folders
for input_folder in input_folders:
    # Create the output folder for the current input folder if it doesn't exist
    output_folder = os.path.join(OUTPUT_BASE_DIR, os.path.basename(input_folder) + "_results")
    os.makedirs(output_folder, exist_ok=True)

    # List for the number of detected nuclei per image
    count_nuclei_list = []

    # Iterate over all images in the current input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Check if it's a supported image format
            image_path = os.path.join(input_folder, filename)

            # Read the test image
            frame = cv2.imread(image_path)
            H, W, _ = frame.shape

            results = model(frame)[0]

            count_nuclei = 0  # Counter for the detected nuclei

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if score > threshold:
                    count_nuclei += 1
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Save the resulting image in the output folder
            output_path = os.path.join(output_folder, f'result_{filename}')
            cv2.imwrite(output_path, frame)

            # Add the number of detected nuclei to the list
            count_nuclei_list.append((filename, count_nuclei))

    # Add the nuclei data of the current folder list to the overall list
    all_count_nuclei_list.extend(count_nuclei_list)

    # Create a DataFrame from the list and save the Excel file for the current folder
    df = pd.DataFrame(count_nuclei_list, columns=['Filename', 'Nuclei Count'])
    excel_output_path = os.path.join(output_folder, 'nuclei_count.xlsx')
    df.to_excel(excel_output_path, index=False)

# Create a DataFrame from the overall list and save the combined Excel file
all_df = pd.DataFrame(all_count_nuclei_list, columns=['Filename', 'Nuclei Count'])
all_excel_output_path = os.path.join(OUTPUT_BASE_DIR, 'combined_nuclei_count.xlsx')
all_df.to_excel(all_excel_output_path, index=False)

# Output the number of detected nuclei for each folder
print("Detection for all folders completed.")
print(f"The nuclei data has been saved in the Excel files.")
print(f"Combined Excel file: {all_excel_output_path}")
