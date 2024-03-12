# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:13:48 2024

@author: saarnij4
"""

import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# Load fixations data from fixations.csv
fixations_data = pd.read_csv('C:/Users/saarnij4/Downloads/Pilot_MARKER-MAPPER_Group_csv (1)/fixations.csv')

# Filter the fixations data to include only rows where "fixation detected on surface" is True
fixations_data = fixations_data[fixations_data['fixation detected on surface']]

# Define mapping of recording IDs to participant numbers
recording_id_to_participant = {
    "cfaeddc3-5130-41a4-bafa-81c504a001be": 1,
    "7d3da7f7-0f79-4e41-8e5c-55330f442cf2": 4,
    "27618aa8-2b29-4210-b17a-7b8cc036579e": 3,
    "4f8880b3-294a-4ab6-80a7-fb860743eae3": 2,
    #"8bd9d4da-3958-4ae6-980c-4994541082ee": 5,
}

# Map recording IDs to participant numbers
fixations_data['participant_id'] = fixations_data['recording id'].map(recording_id_to_participant)
#fixations_data = fixations_data[fixations_data['participant_id'] == 3].head(100)

# Define function to flip coordinates for participant 3 and 4
def flip_coordinates(data):
    if data['participant_id'] in [3, 4]:
        data['fixation x [normalized]'] = 1 - data['fixation x [normalized]']
        data['fixation y [normalized]'] = 1 - data['fixation y [normalized]']
    return data

# Apply flipping of coordinates
fixations_data = fixations_data.apply(flip_coordinates, axis=1)

# Group fixations data by participant ID and count the number of fixations per participant
fixations_per_participant = fixations_data.groupby('participant_id').size().reset_index(name='fixations_count')

# Print the number of fixations per participant
print(fixations_per_participant)

# Plot fixations for each participant
plt.figure(figsize=(12, 8))
for participant_id, group in fixations_data.groupby('participant_id'):
    # Remove outliers based on x and y coordinates
    group = group[(group['fixation x [normalized]'] >= 0) & (group['fixation x [normalized]'] <= 1)]
    group = group[(group['fixation y [normalized]'] >= 0) & (group['fixation y [normalized]'] <= 1)]
    plt.scatter(group['fixation x [normalized]'], group['fixation y [normalized]'], label=f"Participant {participant_id}", alpha=0.5)

plt.title('Fixations for Each Participant (Filtered)')
plt.xlabel('Fixation X (Normalized)')
plt.ylabel('Fixation Y (Normalized)')
plt.legend()
plt.grid(True)
plt.show()


""" Shared fixation condition within +/-2 second time window. Proximity threshold of 0.2 seems appropriate according to the visualization? """


def calculate_shared_fixations(fixation_df, proximity_threshold):
    # Create a list to store overlapping information
    overlap_data = []

    # Sort fixation_df by timestamp
    fixation_df = fixation_df.sort_values(by='start timestamp [ns]')

    # Iterate through unique start timestamps in fixation_df
    for start_timestamp in fixation_df['start timestamp [ns]'].unique():
        # Define the time window for the current start timestamp with a x +/- seconds variation
        window_start = start_timestamp - 0.5e9
        window_end = start_timestamp + 0.5e9

        # Extract fixations within the time window and where fixation detected on surface is True
        fixations = fixation_df[(fixation_df['start timestamp [ns]'] >= window_start) &
                                 (fixation_df['start timestamp [ns]'] <= window_end) &
                                 (fixation_df['fixation detected on surface'])]

        # Check if at least two different recording ids are found in the time window
        if fixations['recording id'].nunique() >= 2:
            # Check for shared fixations based on proximity threshold
            #shared_fixations = []
            shared_fixations, participant_ids = [], []
            for i, fix1 in fixations.iterrows():
                for j, fix2 in fixations.iterrows():
                    if j <= i:
                        continue
                    if fix1['recording id'] != fix2['recording id']:
                        distance = ((fix1['fixation x [normalized]'] - fix2['fixation x [normalized]']) ** 2 +
                                    (fix1['fixation y [normalized]'] - fix2['fixation y [normalized]']) ** 2) ** 0.5
                        
                        if distance <= proximity_threshold:
                            shared_fixations.append(fix1['recording id'])
                            shared_fixations.append(fix2['recording id'])
                            participant_ids.append(fix1["participant_id"])
                            participant_ids.append(fix2["participant_id"])
                            
                            # def within_same_box(fix1, fix2, proximity_threshold):
                            #     if abs(fix1['fixation x [normalized]'] - fix2['fixation x [normalized]']) < proximity_threshold and abs(fix1['fixation y [normalized]'] - fix2['fixation y [normalized]']) < proximity_threshold:
                            #         return True
                            #     return False
                            
                            # if within_same_box(fix1, fix2, proximity_threshold):
                            #     shared_fixations.append(fix1['recording id'])
                            #     shared_fixations.append(fix2['recording id'])
                            #     participant_ids.append(fix1["participant_id"])
                            #     participant_ids.append(fix2["participant_id"])
                            
            if shared_fixations:
                start_time = pd.to_datetime(fixations['start timestamp [ns]'].min())
                end_time = pd.to_datetime(fixations['end timestamp [ns]'].max())
                overlap_data.append({
                    'recording id': shared_fixations,
                    'fixation True start timestamp': start_time,
                    'fixation True end timestamp': end_time,
                    'participant id': participant_ids,
                })

    # Check if there is any overlapping data
    if overlap_data:
        # Convert the list to a DataFrame
        overlap_df = pd.DataFrame(overlap_data)

        # Function to calculate duration in milliseconds
        def calculate_duration(row):
            return (row['fixation True end timestamp'] - row['fixation True start timestamp']).total_seconds() * 1000

        # Apply the function to calculate fixation interval duration
        overlap_df['Fixation Interval Duration (ms)'] = overlap_df.apply(calculate_duration, axis=1)

        # Function to calculate the number of shared fixation participants
        def calculate_shared_participants(row):
            return len(set(row['participant id']))

        # Apply the function to calculate the number of shared fixation participants
        overlap_df['Number of Shared Fixation Participants'] = overlap_df.apply(calculate_shared_participants, axis=1)

        # Group by 'Number of Shared Fixation Participants' and calculate the total fixation interval duration for each group
        result_df = overlap_df.groupby('Number of Shared Fixation Participants').agg(
            Total_Fixation_Interval_Duration=('Fixation Interval Duration (ms)', 'sum'),
            Total_Number_of_Interval=('Fixation Interval Duration (ms)', 'count')
        ).reset_index()

        # Convert the total fixation interval duration to seconds
        result_df['Fixation Duration (s)'] = result_df['Total_Fixation_Interval_Duration'] / 1000

        # Rename the columns for clarity
        result_df = result_df.rename(columns={'Number of Shared Fixation Participants': 'Total_Number_of_Participants'})

        return result_df
    else:
        print('No overlapping data found')


# Assuming fixation_df is your dataframe
proximity_threshold = 0.2
result_df = calculate_shared_fixations(fixations_data, proximity_threshold)
print(result_df)



