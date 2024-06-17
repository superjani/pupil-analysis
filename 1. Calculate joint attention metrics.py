
import pandas as pd
import numpy as np

# Load fixations data from fixations.csv
fixations_data = pd.read_csv('.csv')

# Filter the fixations data to include only rows where "fixation detected on surface" is True
fixations_data = fixations_data[fixations_data['fixation detected on surface']]

# Filter out fixations with duration less than 50ms or more than 2000ms
fixations_data = fixations_data[(fixations_data['duration [ms]'] >= 50) & (fixations_data['duration [ms]'] <= 2000)]

# Read the text file containing the mapping of recording IDs to participant numbers
mapping_file_path = "recording_id_to_participant_number.txt"
mapping_df = pd.read_csv(mapping_file_path, sep=', ', engine='python')

# Create a dictionary from the mapping DataFrame with "Recording ID" as keys and "Participant" as values
recording_id_to_participant = dict(zip(mapping_df['recording id'], mapping_df['Participant ID']))

# Now, you can use the recording_id_to_participant dictionary to map recording IDs to participant numbers
# For example:
fixations_data['participant_id'] = fixations_data['recording id'].map(recording_id_to_participant)

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

# Visualize the location of fixations in the AOI 

import matplotlib.pyplot as plt

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

def calculate_shared_fixations(fixation_df, proximity_threshold):
    # Create a list to store overlapping information
    overlap_data = []

    # Sort fixation_df by timestamp
    fixation_df = fixation_df.sort_values(by='start timestamp [ns]').reset_index()

    # Iterate through unique start timestamps in fixation_df
    for i, fix1 in fixation_df.iterrows():
        # Define the time window for the current start timestamp with a x +/- seconds variation
        start_timestamp = fix1['start timestamp [ns]']
        window_start = start_timestamp
        window_end = start_timestamp + 1.0e9

        # Extract fixations within the time window and where fixation detected on surface is True
        fixations = fixation_df[(fixation_df['start timestamp [ns]'] >= window_start)]
        fixations = fixations[fixations['start timestamp [ns]'] <= window_end]
        fixations = fixations[fixations['fixation detected on surface']]

        # Check if at least two different recording ids are found in the time window
        if fixations['recording id'].nunique() >= 2:
            # Check for shared fixations based on proximity threshold
            shared_fixations, participant_ids = [], []
            x_locations = []
            y_locations = []
            start_times = []
            end_times = []
            for j, fix2 in fixations.iterrows():
                    # if j <= i:
                    #     continue
                    if fix1['recording id'] != fix2['recording id']:
                        distance = ((fix1['fixation x [normalized]'] - fix2['fixation x [normalized]']) ** 2 +
                                    (fix1['fixation y [normalized]'] - fix2['fixation y [normalized]']) ** 2) ** 0.5
                        
                        if distance <= proximity_threshold:
                            if shared_fixations == []:
                                shared_fixations.append(fix1['recording id'])
                                participant_ids.append(fix1["participant_id"])
                                x_locations.append(fix1['fixation x [normalized]'])
                                y_locations.append(fix1['fixation y [normalized]'])
                                start_times.append(fix1['start timestamp [ns]'])
                                end_times.append(fix1['end timestamp [ns]'])
                            shared_fixations.append(fix2['recording id'])
                            participant_ids.append(fix2["participant_id"])
                            x_locations.append(fix2['fixation x [normalized]'])
                            y_locations.append(fix2['fixation y [normalized]'])
                            start_times.append(fix2['start timestamp [ns]'])
                            end_times.append(fix2['end timestamp [ns]'])
                            
            if shared_fixations:
                start_time = pd.to_datetime(min(start_times))
                end_time = pd.to_datetime(max(end_times))
                data = {
                    'recording id': shared_fixations,
                    'fixation True start timestamp': start_time,
                    'fixation True end timestamp': end_time,
                    'participant id': participant_ids,
                    'mean fixation x': np.array(x_locations).mean(),
                    'mean fixation y': np.array(y_locations).mean()
                }
                    
                for k in range(len(participant_ids)):
                    p = participant_ids[k]
                    data[f"participant {p} start time"] = pd.to_datetime(start_times[k])
                    data[f"participant {p} end time"] =  pd.to_datetime(end_times[k])
            
                overlap_data.append(data)

    # Check if there is any overlapping data
    if overlap_data:
        
        # Convert the list to a DataFrame
        overlap_df = pd.DataFrame(overlap_data)
        
        # Check if the set of participants in a shared fixation entry (row) is a subset of the participants in the previous entry. 
        # This check helps to identify cases where one shared fixation is a subset of another, indicating redundancy or overlap.
        def is_participant_subset(row):
            if  row["shifted id"] is None:
                return False
            return set(row["shifted id"]) <= set(row["participant id"])
  
        overlap_df["shifted id"] = overlap_df["participant id"].shift()
        dublicated = overlap_df.apply(is_participant_subset, axis=1)
        # Overlapping time comparison
        dublicated = dublicated & (overlap_df["fixation True end timestamp"] == overlap_df.shift()["fixation True end timestamp"])
        # Excluding the duplicated data 
        overlap_df = overlap_df[~dublicated]
        
        # Create sets to store end times for each participant
        end_times_sets = [{}, {}, {}, {}]
        
        # Create a list to store indices of rows to be removed
        rows_to_remove = []
        
        # Iterate over the DataFrame rows
        for index, row in overlap_df.iterrows():
            # Check for duplicates in each participant's end time
            for participant_id in range(1, 5):
                end_time = row[f'participant {participant_id} end time']
                # Check if end time is NaN, which indicates missing data
                if pd.isnull(end_time):
                    # If end time is missing, skip this row
                    continue
                # Check if end time is already in the set
                if end_time in end_times_sets[participant_id - 1]:
                    rows_to_remove.append(index)
                else:
                    # If not, add this end time to the set
                    end_times_sets[participant_id - 1][end_time] = True
        
        # Remove rows marked for removal
        if rows_to_remove:
            print(f"Removing {len(rows_to_remove)} rows with duplicate end times")
            overlap_df.drop(index=rows_to_remove, inplace=True)
        else:
            print("No rows found with duplicate end times")

        # Function to calculate duration in milliseconds
        def calculate_duration(row):
            return (row['fixation True end timestamp'] - row['fixation True start timestamp']).total_seconds() * 1000

        # Apply the function to calculate joint attention duration, which is not equal to the sum of individual fixation durations
        overlap_df['Joint Attention Duration (ms)'] = overlap_df.apply(calculate_duration, axis=1)
        # Participant fixation duration in joint attention. However, only one fixation is included. 
        for _, p in recording_id_to_participant.items():
            if f"participant {p} end time" in overlap_df.columns:
                overlap_df[f'Participant {p} Fixation Interval Duration (ms)'] = overlap_df[f"participant {p} end time"] -  overlap_df[f"participant {p} start time"]

        # Function to calculate the number of joint attention participants
        def calculate_shared_participants(row):
            return len(set(row['participant id']))

        # Apply the function to calculate the number of shared fixation participants
        overlap_df['Number of Shared Fixation Participants'] = overlap_df.apply(calculate_shared_participants, axis=1)

        # Group by 'Number of Shared Fixation Participants' and calculate the total fixation interval duration for each group
        result_df = overlap_df.groupby('Number of Shared Fixation Participants').agg(
            Total_Joint_Attention_Duration_ms=('Joint Attention Duration (ms)', 'sum'),
            Total_Number_of_Intervals=('Joint Attention Duration (ms)', 'count'),
            Mean_Joint_Attention_Duration_ms=('Joint Attention Duration (ms)', 'mean')
        ).reset_index()

        # Convert the total fixation interval duration to seconds
        result_df['Joint Attention Duration (s)'] = result_df['Total_Joint_Attention_Duration_ms'] / 1000

        # Rename the columns for clarity
        result_df = result_df.rename(columns={'Number of Shared Fixation Participants': 'Total_Number_of_Participants'})

        overlap_df.drop(["shifted id"], inplace = True, axis=1)

        return result_df, overlap_df
    else:
        print('No overlapping data found')

# Assuming fixation_df is your dataframe
proximity_threshold = 0.2
result_df, overlap_df = calculate_shared_fixations(fixations_data, proximity_threshold)

# Save the overlap_df DataFrame as a CSV file
overlap_df.to_csv('.csv', index=False)

print(result_df)

result_df.to_csv('.csv', index=False)
