# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:53:54 2024

@author: akbar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
from time import sleep
from IPython.display import Image, display
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EfficientNetV2_path = r"C:/Users/Akbar/OneDrive/EfficientNetV2_S&SGD"
first_model= torch.load(EfficientNetV2_path, map_location=torch.device('cpu'))
first_model.eval()
ResNet18_flow_path = r"C:/Users/Akbar/OneDrive/Mamba_model2"
third_model= torch.load(ResNet18_flow_path, map_location=torch.device('cpu'))
third_model.eval()

def predict_fire(model, image):
    resized_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(resized_image)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    def preprocess_image(image):
        image = transform(image)
        image = image.unsqueeze(0)
        return image.to(device)
    test_image = preprocess_image(image)
    with torch.no_grad():
        test_image = test_image.to(device)
        outputs = model(test_image)
        probabilities = F.softmax(outputs, dim=1)
        predicted_probability = probabilities[0][0].item()
    return predicted_probability

def calculate_optical_flow(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_image = flow_to_image(flow)
    return flow_image
def flow_to_image(flow):
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_image = np.uint8(mag)
    return flow_image

def predict_flow(model, image):
    resized_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(resized_image)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    def preprocess_image(image):
        image = transform(image)
        image = image.unsqueeze(0)
        return image.to(device)
    test_image = preprocess_image(image)
    with torch.no_grad():
        test_image = test_image.to(device)
        outputs = model(test_image)
        probabilities = F.softmax(outputs, dim=1)
        predicted_probability, predicted_class = probabilities[0][1].item(), torch.argmax(probabilities, dim=1).item()
    return predicted_probability, predicted_class

def get_video_names(folder_path):
    video_names = []
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Get the list of files in the folder and sort them
        files = sorted(os.listdir(folder_path))

        # Iterate over the sorted files in the folder
        for filename in files:
            # Check if the file is a video (customize this condition based on your file extensions)
            if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                # Remove the extension and append to the list
                video_name_without_extension = os.path.splitext(filename)[0]
                video_names.append(video_name_without_extension)
    else:
        print(f"The folder {folder_path} does not exist.")

    return video_names

# Replace 'path_to_folder' with the actual path to your folder containing videos
folder_path = r"C:/Users/Akbar/OneDrive/Trial Videos"
video_names_list = get_video_names(folder_path)

# Display the list of video names without extensions in the order they appear in the folder
print("List of video names:")
for video_name in video_names_list:
    print(video_name)

def create_directory_structure(source_folder, target_folder_first, target_folder_flow):
    # Ensure target folders exist
    os.makedirs(target_folder_first, exist_ok=True)
    os.makedirs(target_folder_flow, exist_ok=True)

    # Iterate over files in the source folder
    for filename in os.listdir(source_folder):
        print(f"Processing file: {filename}")  # Debug print
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more video formats if needed
            # Extract name without extension
            name, _ = os.path.splitext(filename)
            print(f"Extracted name: {name}")  # Debug print

            # Create subfolders in target_folder_first
            first_folder_path = os.path.join(target_folder_first, name)
            os.makedirs(first_folder_path, exist_ok=True)
            os.makedirs(os.path.join(first_folder_path, 'A'), exist_ok=True)
            os.makedirs(os.path.join(first_folder_path, 'B'), exist_ok=True)
            os.makedirs(os.path.join(first_folder_path, 'Frames'), exist_ok=True)
            print(f"Created folders in {target_folder_first} for: {filename}")  # Debug print

            # Create subfolders in target_folder_flow
            flow_folder_path = os.path.join(target_folder_flow, name)
            os.makedirs(flow_folder_path, exist_ok=True)
            os.makedirs(os.path.join(flow_folder_path, 'A'), exist_ok=True)
            os.makedirs(os.path.join(flow_folder_path, 'B'), exist_ok=True)
            os.makedirs(os.path.join(flow_folder_path, 'Flow'), exist_ok=True)
            os.makedirs(os.path.join(flow_folder_path, 'Frames'), exist_ok=True)
            print(f"Created folders in {target_folder_flow} for: {filename}")  # Debug print

# Example usage
source_folder = r"C:/Users/Akbar/OneDrive/Trial Videos"
target_folder_first = r"C:/Users/Akbar/OneDrive/Trial Videos/First"
target_folder_flow = r"C:/Users/Akbar/OneDrive/Trial Videos/Flow"

create_directory_structure(source_folder, target_folder_first, target_folder_flow)

base_path =  r"C:/Users/Akbar/OneDrive/Trial Videos/First"
base_path_flow = r"C:/Users/Akbar/OneDrive/Trial Videos/Flow"

threshold_alarm_fire = 0.95
threshold_alarm_fire_model = 0.75
input_height = 224
input_width = 224
red = (0, 0, 255)
green = (0, 255, 0)

def process_video(skip_frames):
    frame_number = 1
    all_frames = []

    Fire_model = []
    Motion_model = []

    Fire_temp= []
    Motion_temp= []

    while frame_number <= 80:
        ret, frame = name.read()
        if not ret:
            break
        all_frames.append(frame)

        frame_number += 1

    frame_number = 1

    while frame_number <= len(all_frames) - skip_frames:

        image1 = all_frames[frame_number - 1]
        image2 = all_frames[frame_number - 1 + skip_frames]

        image1 = cv2.resize(image1, (1920, 1080))
        image2 = cv2.resize(image2, (1920, 1080))

        img_draw = image1.copy()
        img_draw_flow = image1.copy()


        for m in range(5):
            for n in range(9):
                original_image1 = cv2.resize(image1[m * 180:m * 180 + 359, n * 180 + 60:n * 180 + 359 + 60, :], (input_height, input_width))
                original_image2 = cv2.resize(image2[m * 180:m * 180 + 359, n * 180 + 60:n * 180 + 359 + 60, :], (input_height, input_width))

                results_Fire = predict_fire(first_model, original_image1)
                #Fire_temp = []
                if results_Fire > threshold_alarm_fire:
                    filename_img1 = f'{folder_name}_frame_{frame_number}_block_{m}_{n}.png'
                    directory_path = f'{base_path}/{folder_name}/A'
                    file_path = os.path.join(directory_path, filename_img1)
                    #cv2.imwrite(file_path, original_image1)
                    filename_img2 = f'{folder_name}_frame_{frame_number + skip_frames}_block_{m}_{n}.png'
                    directory_path = f'{base_path}/{folder_name}/B'
                    file_path = os.path.join(directory_path, filename_img2)
                   # cv2.imwrite(file_path, original_image2)
                    Fire_model.append('fire')
                    Fire_temp.append('fire')
                    color = red
                    cv2.rectangle(img_draw, (n * 180 + 60, m * 180), (n * 180 + 360 + 60, m * 180 + 360), red, 2)
                    cv2.putText(img_draw, str(results_Fire * 100)[0:5] + "%", (n * 180 + 60, m * 180 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    print(f"Processed frame {frame_number} from the first model, Block ({m},{n})")
                    print("Number of fire detected blocks using only the first model:", len(Fire_temp))

                else:
                    color = green
                    cv2.rectangle(img_draw, (n * 180 + 60, m * 180), (n * 180 + 360 + 60, m * 180 + 360), green, 2)
                    cv2.putText(img_draw, str(results_Fire * 100)[0:5] + "%", (n * 180 + 60, m * 180 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)



#-------------------------------------------------------------------------------------------------------------------------

                if results_Fire > threshold_alarm_fire:

                    flow_image = calculate_optical_flow(original_image1, original_image2)
                    directory_path = f'{base_path_flow}/{folder_name}/Flow'
                    filename = f'{folder_name}_flow_frame_{frame_number}_block_{m}_{n}.png'
                    file_path = os.path.join(directory_path, filename)
                    #cv2.imwrite(file_path, flow_image)

                    results_flow, predicted_class_flow = predict_flow(third_model, flow_image)

                    # Motion_temp = []
                    print(results_flow)
                    if results_flow > threshold_alarm_fire:

                        filename_img1 = f'{folder_name}_frame_{frame_number}_block_{m}_{n}.png'
                        directory_path = f'{base_path_flow}/{folder_name}/A'
                        file_path = os.path.join(directory_path, filename_img1)
                        #cv2.imwrite(file_path, original_image1)

                        filename_img2 = f'{folder_name}_frame_{frame_number + skip_frames}_block_{m}_{n}.png'
                        directory_path = f'{base_path_flow}/{folder_name}/B'
                        file_path = os.path.join(directory_path, filename_img2)
                        #cv2.imwrite(file_path, original_image2)

                        Motion_model.append('fire')
                        Motion_temp.append('fire')
                        color = red
                        cv2.rectangle(img_draw_flow, (n * 180 + 60, m * 180), (n * 180 + 360 + 60, m * 180 + 360), red, 2)
                        cv2.putText(img_draw_flow, str(results_flow * 100)[0:5] + "%", (n * 180 + 60, m * 180 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        print(f"Processed frame {frame_number} from the motion model, Block ({m},{n})")
                        print("Number of fire detected blocks using the first model and motion model:", len(Motion_temp))

                    else:
                        color = green
                        cv2.rectangle(img_draw_flow, (n * 180 + 60, m * 180), (n * 180 + 360 + 60, m * 180 + 360), green, 2)
                        cv2.putText(img_draw_flow, str(results_flow * 100)[0:5] + "%", (n * 180 + 60, m * 180 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                else:
                    color = green
                    cv2.rectangle(img_draw_flow, (n * 180 + 60, m * 180), (n * 180 + 360 + 60, m * 180 + 360), green, 2)
                    cv2.putText(img_draw_flow, str(results_Fire * 100)[0:5] + "%", (n * 180 + 60, m * 180 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        output_directory = f'{base_path}/{folder_name}/Frames'
        frame_filepath = os.path.join(output_directory, f'{folder_name}_final_frame_{frame_number}.png')
        #cv2.imwrite(frame_filepath, img_draw)
        cv2_imshow(img_draw)
        cv2.waitKey(40)
        output_directory = f'{base_path_flow}/{folder_name}/Frames'
        frame_filepath = os.path.join(output_directory, f'{folder_name}_final_frame_{frame_number}.png')
        #cv2.imwrite(frame_filepath, img_draw_flow)
        cv2_imshow(img_draw_flow)
        cv2.waitKey(40)
        print(f"Frame {frame_number} has been processed")
        frame_number += 1

    print('--------------------------------------------------------------------------------------------------------------------------')

    print(f"Total Number of blocks in the entire video: {(frame_number*45)}")

    print(f"Total Number of Fire detected blocks in the entire video using using the first model: \033[1m{len(Fire_model)}\033[0m")
    print(f"Total Number of Fire detected blocks in the entire video using using the first model + motion estimation: \033[1m{len(Motion_model)}\033[0m")

    print('--------------------------------------------------------------------------------------------------------------------------')

    if len(Fire_model) == 0:

        print("First model did not predict fire")

    else:

        print(f"True Detection Rate using the first model + motion estimation: \033[1m{((len(Motion_model))/(len(Fire_model)))*100}\033[0m")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Ensure that the list index is within bounds
start_index =0
end_index = 7

# Print the length of the video names list for debugging
print(f"Number of videos found: {len(video_names_list)}")

# Ensure the end_index does not exceed the length of the video_names_list
if end_index >= len(video_names_list):
    end_index = len(video_names_list) - 1

for i in range(start_index, end_index + 1):
    folder_name = video_names_list[i]
    print(f"\033[1;91mProcessing video Number {i}: {folder_name}\033[0m")
    name = f"C:/Users/Akbar/OneDrive/Trial Videos/{folder_name}.mp4"
    name = cv2.VideoCapture(name)
    process_video(1)
    print(f"\033[91mVideo processing complete for: {folder_name}\033[0m\n")
    print('--------------------------------------------------------------------------------------------------------------------------')

