import pandas as pd
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Define the root folder path
root_folder_path = ''

# Read the text file containing the mapping of recording IDs to participant numbers
mapping_file_path = "/recording_id_to_participant_number.txt"
mapping_df = pd.read_csv(mapping_file_path, sep=', ', engine='python')

# Create a dictionary from the mapping DataFrame with "Recording ID" as keys and "Participant" as values
recording_id_to_participant = dict(zip(mapping_df['recording id'], mapping_df['Participant ID']))

THRESHOLD_DISTANCE = 10
THRESHOLD_TIME = 2

# Function to process each fixations.csv file
def process_fixations_file(file_path):
    fixations_data = pd.read_csv(file_path)
    
    # Extract AOI name from the directory path
    aoi_name = os.path.basename(os.path.dirname(file_path))
    fixations_data['AOI'] = aoi_name

    # Extract Group name from the directory path
    directory_path = os.path.dirname(file_path)
    group_name = os.path.basename(os.path.dirname(directory_path))
    
    # Map recording IDs to participant numbers
    fixations_data['participant_id'] = fixations_data['recording id'].map(recording_id_to_participant)

    # Check if both participants (3 and 4) are present
    participants_present = fixations_data['participant_id'].unique()
    
    # Flip coordinates for participants 3 and 4 if they are present, but exclude Screen
    if 3 in participants_present:
        mask = (fixations_data['participant_id'] == 3) & ~fixations_data['AOI'].str.contains('screen', case=False, na=False)
        fixations_data.loc[mask, ['fixation x [normalized]', 'fixation y [normalized]']] = 1 - fixations_data.loc[mask, ['fixation x [normalized]', 'fixation y [normalized]']]

    if 4 in participants_present:
        mask = (fixations_data['participant_id'] == 4) & ~fixations_data['AOI'].str.contains('screen', case=False, na=False)
        fixations_data.loc[mask, ['fixation x [normalized]', 'fixation y [normalized]']] = 1 - fixations_data.loc[mask, ['fixation x [normalized]', 'fixation y [normalized]']]

    #### Do we want to filter out fixations under 0 or over 1? Pupil Invisible counts all 
    # fixations_data = fixations_data[
    #     (fixations_data['fixation x [normalized]'] >= 0) & (fixations_data['fixation x [normalized]'] <= 1) &
    #     (fixations_data['fixation y [normalized]'] >= 0) & (fixations_data['fixation y [normalized]'] <= 1)]

    # Scale the AOI coordinates based on their physical size (cm) (varmista, että x on aina samansuuntainen)
    scale_map = {
        "ScreenGroupWork": [67.0, 43.0],
        "TableGroupWork": [78.0, 45.0], 
        "P1GroupWork": [38.0, 27.0], 
        "P2GroupWork": [38.0, 27.0], 
        "P3GroupWork": [38.0, 27.0], 
        "P4GroupWork": [38.0, 27.0], 
        "Objects1GroupWork": [21.7, 41.0], 
        "Objects2GroupWork": [21.7, 41.0], 
        "Side1GroupWork": [25.0, 21.0], 
        "Side2GroupWork": [25.0, 21.0],
        "TableIndividualWork": [78.0, 45.0],
        "ScreenIndividualWork": [67.0, 43.0],
        "P1IndividualWork": [38.0, 27.0], 
        "P2IndividualWork": [38.0, 27.0], 
        "P3IndividualWork": [38.0, 27.0], 
        "P4IndividualWork": [38.0, 27.0], 
        "Objects1IndividualWork": [21.7, 41.0], 
        "Objects2IndividualWork": [21.7, 41.0], 
        "Side1IndividualWork": [25.0, 21.0], 
        "Side2IndividualWork": [25.0, 21.0]
        } 
    
    for AOI, scales in scale_map.items():
        mask = fixations_data["AOI"] == AOI
        if np.any(mask):    
            fixations_data.loc[mask, 'fixation x [normalized]'] = scales[0] * fixations_data.loc[mask, 'fixation x [normalized]']
            fixations_data.loc[mask, 'fixation y [normalized]'] = scales[1] * fixations_data.loc[mask, 'fixation y [normalized]']

    # Identify visits per participant
    fixations_data['visit_id'] = fixations_data.groupby('participant_id')['fixation detected on surface'].transform(lambda x: (x != x.shift()).cumsum())

    # Filter out fixations with duration less than 50ms or more than 2000ms and fixations not detected on surface
    filtered_fixations = fixations_data[
        (fixations_data['duration [ms]'] >= 50) &
        (fixations_data['duration [ms]'] <= 2000) &
        (fixations_data['fixation detected on surface'])
    ].copy()
    
    filtered_fixations['start_timestamp'] = filtered_fixations['start timestamp [ns]'].apply(lambda x: pd.to_datetime(x, unit='ns', errors='coerce'))
    filtered_fixations['end_timestamp'] = filtered_fixations['end timestamp [ns]'].apply(lambda x: pd.to_datetime(x, unit='ns', errors='coerce'))

    # Drop rows with NaT (Not a Time) values in 'start_timestamp' or 'end_timestamp'
    filtered_fixations.dropna(subset=['start_timestamp', 'end_timestamp'], inplace=True)
    
    # Ensure all timestamps are datetime64[ns] type
    filtered_fixations['start_timestamp'] = pd.to_datetime(filtered_fixations['start_timestamp'], errors='coerce')
    filtered_fixations['end_timestamp'] = pd.to_datetime(filtered_fixations['end_timestamp'], errors='coerce')
    

    # Compute total duration in minutes
    total_duration_minutes = (filtered_fixations['end_timestamp'].max() - filtered_fixations['start_timestamp'].min()).total_seconds() / 60

    # Calculate metrics for each visit
    visit_metrics = filtered_fixations.groupby(['participant_id', 'visit_id']).agg(
        visit_duration=('duration [ms]', 'sum'),
        num_fixations=('duration [ms]', 'size'),
        mean_fixation_duration=('duration [ms]', 'mean'),
        std_fixation_duration=('duration [ms]', 'std'),
        start_timestamp=('start_timestamp', 'min'),
        end_timestamp=('end_timestamp', 'max'),
        mean_fixation_x=('fixation x [normalized]', 'mean'),
        mean_fixation_y=('fixation y [normalized]', 'mean'),
        std_fixation_x=('fixation x [normalized]', 'std'), 
        std_fixation_y=('fixation y [normalized]', 'std')
    ).reset_index()
    

    # Add AOI and placeholder columns
    visit_metrics['AOI'] = aoi_name
    visit_metrics['Joint Attention'] = False
    visit_metrics['Initiation'] = False
    visit_metrics['Gaze Follower'] = False
    visit_metrics['Total duration of the Section (min)'] = total_duration_minutes
    visit_metrics["Group"] = group_name
    
    # List of AOI prefixes
    aoi_prefixes = ["Screen", "Table", "P1", "P2", "P3", "P4", "Objects1", "Objects2", "Side1", "Side2"]

    # Function to remove AOI prefix and extract section name
    def extract_section(aoi_name):
        for prefix in aoi_prefixes:
            if aoi_name.startswith(prefix):
                return aoi_name[len(prefix):]
        return aoi_name

    # Apply the function to create a new column for the section
    visit_metrics['Section'] = visit_metrics['AOI'].apply(extract_section)
        
    visit_metrics = visit_metrics.rename(columns={'participant_id': 'Participant ID'})
    visit_metrics['Participant ID'] = visit_metrics['Participant ID'].astype(str)

    # Iterate over rows of the DataFrame
    for index, row in visit_metrics.iterrows():
        # Extract the last character from the 'Group' column
        group_suffix = row['Group'][-1]
        
        # Read the participant ID as a string
        participant_id_str = row['Participant ID']
        
        # Convert to integer if the value is an integer-like string
        try:
            participant_id = int(participant_id_str)
        except ValueError:
            # Handle the case where the value is not a valid integer
            participant_id = int(float(participant_id_str))
        
        # Update the 'Participant ID' column by concatenating 'G' with the group suffix and 'P' with the participant ID
        visit_metrics.at[index, 'Participant ID'] = 'G{}P{}'.format(group_suffix, participant_id)
    
    def check_joint_attention_vectorized(df):
        
        # Extract the participant IDs and coordinates
        participant_ids = df['Participant ID'].values
        mean_fixation_x = df['mean_fixation_x'].values
        mean_fixation_y = df['mean_fixation_y'].values
    
        # Create a distance matrix
        x_diff = mean_fixation_x[:, None] - mean_fixation_x
        y_diff = mean_fixation_y[:, None] - mean_fixation_y
        distance_matrix = np.sqrt(x_diff ** 2 + y_diff ** 2)
    
        # Create a time proximity matrix using numpy operations
        start_times = df['start_timestamp'].values
        end_times = df['end_timestamp'].values
    
        # Vectorized time proximity calculation
        start_diff = np.abs(start_times[:, None] - start_times)
        end_diff = np.abs(end_times[:, None] - end_times)
        time_proximity_matrix = ((start_diff <= np.timedelta64(THRESHOLD_TIME, 's')) |
                                 (end_diff <= np.timedelta64(THRESHOLD_TIME, 's')) |
                                 ((start_times[:, None] <= end_times) & 
                                  (start_times <= end_times[:, None])))
    
        # Create a participant comparison matrix: True if fixations are from different participants
        different_participant_matrix = participant_ids[:, None] != participant_ids
    
        # Identify joint attention ----> Mitä vastaa AOI 
        joint_attention_matrix = (distance_matrix <= THRESHOLD_DISTANCE) & time_proximity_matrix & different_participant_matrix
            
        # Use numpy's any() to determine if any joint attention occurs for each participant
        joint_attention_flags = np.any(joint_attention_matrix, axis=1)
        
        return joint_attention_matrix, joint_attention_flags

    # Check for joint attention
    joint_attention_matrix, joint_attention_flags = check_joint_attention_vectorized(visit_metrics)
    
    # Use joint attention matrix as an adjency matrix to create graph of events
    # Each participant is a node and lines connect who have joint attention 
    event_graph = nx.from_numpy_array(joint_attention_matrix)
            
    # Add participant id to graph
    nx.set_node_attributes(event_graph, visit_metrics['Participant ID'], 'Participant ID')
    # Calculate unique particpants for each visit
    visit_metrics['Joint Attention Count'] = 0
    for node in event_graph:
        node_degree = event_graph.degree[node]
        if node_degree == 0:
            continue
        neighbor_graph = nx.subgraph(event_graph, event_graph.neighbors(node))
        neighbor_participant_ids = nx.get_node_attributes(neighbor_graph, 'Participant ID').values()
        visit_metrics.loc[node, 'Joint Attention Count'] = len(set((neighbor_participant_ids)))
        #print(f'Node {node} has {node_degree} neighbors with parcipant ids {neighbor_participant_ids}')
        
    # Add participant itself to joint attention count
    visit_metrics['Joint Attention Count'] =  visit_metrics['Joint Attention Count'] + 1
    
    #print(f'Found {nx.number_connected_components(event_graph)} unique event groups from {visit_metrics.shape[0]} visits')
    
    # Label each disconnected subgraph as a separate event group
    visit_metrics['Event ID'] = -1
    for i, subgraph in enumerate(nx.connected_components(event_graph)):
        visit_metrics.loc[list(subgraph), 'Event ID'] = i
    
    visit_metrics['Event Participants'] = -1
    for event_id, participants in visit_metrics[['Event ID', 'Participant ID']].groupby('Event ID').nunique().iterrows():
        visit_metrics.loc[visit_metrics['Event ID'] == event_id, 'Event Participants'] = participants['Participant ID']
    
    visit_metrics['Linked Visits Count'] = 0
    for i, d in list(nx.degree(event_graph)):
        visit_metrics.loc[i, 'Linked Visits Count'] = d
        
    # Assign joint attention flags
    visit_metrics['Joint Attention'] = joint_attention_flags
    
    # Set Initiation and Gaze Follower to False initially
    visit_metrics['Initiation'] = False
    visit_metrics['Gaze Follower'] = False
    
    # Perform the Initiation logic only for rows where Joint Attention is True
    for event_id, event_df in visit_metrics.loc[visit_metrics['Joint Attention'], ['Event ID', 'start_timestamp']].groupby('Event ID'):
        initiator_idx = event_df['start_timestamp'].idxmin()
        visit_metrics.loc[initiator_idx, 'Initiation'] = True
    
    # Calculate Gaze Follower based on Initiation for rows where Joint Attention is True
    visit_metrics.loc[visit_metrics['Joint Attention'], 'Gaze Follower'] = ~visit_metrics['Initiation']

    # # Define unique groups
    # unique_groups = visit_metrics['Group'].unique()
    
    # # Initialize the figure
    # fig, axes = plt.subplots(nrows=1, ncols=len(unique_groups), figsize=(20, 5), sharex=True, sharey=True)
    
    # for idx, group in enumerate(unique_groups):
    #     # Filter the visits for the current group
    #     group_visits = visit_metrics[visit_metrics['Group'] == group]
    
    #     # Sort the visits by time
    #     sorted_visits = group_visits.sort_values(by='start_timestamp')
    
    #     # Create a list of unique participant IDs for the current group
    #     participant_ids = sorted_visits['Participant ID'].unique()
    
    #     # Initialize the cross-recurrence matrix
    #     recurrence_matrix = np.zeros((len(participant_ids), len(participant_ids)))
    #     participant_counts_matrix = np.zeros((len(participant_ids), len(participant_ids)))
    
    #     # Populate the recurrence matrix
    #     for i, participant_1 in enumerate(participant_ids):
    #         for j, participant_2 in enumerate(participant_ids):
    #             if i != j:
    #                 visits_1 = sorted_visits[sorted_visits['Participant ID'] == participant_1]
    #                 visits_2 = sorted_visits[sorted_visits['Participant ID'] == participant_2]
    #                 overlap_count = 0
    #                 for start1, end1 in zip(visits_1['start_timestamp'], visits_1['end_timestamp']):
    #                     for start2, end2 in zip(visits_2['start_timestamp'], visits_2['end_timestamp']):
    #                         if (start1 < end2) and (end1 > start2):
    #                             overlap_count += 1
    #                 recurrence_matrix[i, j] = overlap_count
    
    #     # Populate the participant counts matrix
    #     for i, participant_1 in enumerate(participant_ids):
    #         for j, participant_2 in enumerate(participant_ids):
    #             if i != j:
    #                 common_events = sorted_visits[
    #                     (sorted_visits['Participant ID'] == participant_1) |
    #                     (sorted_visits['Participant ID'] == participant_2)
    #                 ]
    #                 participant_counts = common_events.groupby('Event ID')['Participant ID'].nunique()
    #                 max_participants = participant_counts.max() if not participant_counts.empty else 0
    #                 participant_counts_matrix[i, j] = max_participants
    
    #     # Create a heatmap of the participant counts matrix
    #     sns.heatmap(participant_counts_matrix, annot=True, cmap='viridis', xticklabels=participant_ids, yticklabels=participant_ids, ax=np.ravel(axes)[idx])
    #     np.ravel(axes)[idx].set_title(f'Group {group}')
    #     np.ravel(axes)[idx].set_xlabel('Participant ID')
    #     np.ravel(axes)[idx].set_ylabel('Participant ID')
        
    #     # Adjust the layout and show the plots
    #     plt.tight_layout()
        
    #     # Move the plt.show() outside of the loop
    #     plt.show()

    return visit_metrics

# List to store all DataFrames
dfs = []

# Loop through each group directory
for group_folder in os.listdir(root_folder_path):
    group_folder_path = os.path.join(root_folder_path, group_folder)
    if os.path.isdir(group_folder_path):
        # Loop through each subdirectory in the group folder
        for sub_folder in os.listdir(group_folder_path):
            sub_folder_path = os.path.join(group_folder_path, sub_folder)
            if os.path.isdir(sub_folder_path) and sub_folder.lower().startswith(("table", "screen", "p1", "p3", "p2", "p4", "objects1", "objects2", "side1", "side2")):
                # Process the fixations.csv file
                fixations_file_path = os.path.join(sub_folder_path, 'fixations.csv')
                if os.path.exists(fixations_file_path):
                    processed_df = process_fixations_file(fixations_file_path)
                    dfs.append(processed_df)

# Concatenate all DataFrames
final_df = pd.concat(dfs, ignore_index=True)
#final_df = dfs[0]

# Sort the final DataFrame by GroupNumber and ParticipantNumber
final_df = final_df.sort_values(by=['Group', 'Participant ID', 'start_timestamp']).reset_index(drop=True)

# Re-number the visit_id for each participant
final_df['visit_id'] = final_df.groupby('Participant ID').cumcount() + 1

final_df['start_timestamp'] = pd.to_datetime(final_df['start_timestamp'])
final_df['end_timestamp'] = pd.to_datetime(final_df['end_timestamp'])

initiation_counts = final_df.loc[(final_df['Initiation'] & final_df['Joint Attention']), ['Participant ID', 'Event ID']].groupby('Participant ID').count()
joint_attention_counts = final_df.loc[(final_df['Joint Attention'] & final_df['Joint Attention']), ['Participant ID', 'Event ID']].groupby('Participant ID').count()

# Rename the 'Event ID' column to 'initiation counts'
initiation_counts.rename(columns={'Event ID': 'Initiation counts'}, inplace=True)
joint_attention_counts.rename(columns={'Event ID': 'Joint Attention counts'}, inplace=True)

# Merge the initiation counts back into the original DataFrame
final_df = final_df.merge(initiation_counts, on='Participant ID', how='left')
final_df = final_df.merge(joint_attention_counts, on='Participant ID', how='left')

# Define a function to calculate maximum distance between visits on one event
def max_distance(group):
    if len(group) < 2:
        return 0
    coords = group[['mean_fixation_x', 'mean_fixation_y']].values
    distances = np.sqrt((coords[:, np.newaxis, 0] - coords[np.newaxis, :, 0])**2 + 
                        (coords[:, np.newaxis, 1] - coords[np.newaxis, :, 1])**2)
    return distances.max()

# Define a function to calculate event duration
def event_duration(group):
    start_times = pd.to_datetime(group['start_timestamp'])
    end_times = pd.to_datetime(group['end_timestamp'])
    duration = (end_times.max() - start_times.min()).total_seconds()
    return duration

# Ensure the columns are present before performing the groupby operation
required_columns = ['Event ID', 'Group', 'AOI', 'mean_fixation_x', 'mean_fixation_y', 'visit_id', 'start_timestamp', 'end_timestamp', 'Event Participants', 'Linked Visits Count']
missing_columns = [col for col in required_columns if col not in final_df.columns]

if missing_columns:
    print(f"Error: The following required columns are missing from the DataFrame: {missing_columns}")
else:
    # Group by 'Event ID' and 'AOI'
    grouped = final_df.groupby(['Event ID', 'AOI', 'Group'])

    # Apply the custom functions to the entire group
    custom_metrics = grouped.apply(lambda group: pd.Series({
        'max_distance': max_distance(group),
        'event_duration': event_duration(group)
    }))

    # Apply standard aggregations
    standard_metrics = grouped.agg(
        mean_visit_x=('mean_fixation_x', 'mean'),
        mean_visit_y=('mean_fixation_y', 'mean'),
        std_visit_x=('mean_fixation_x', 'std'),
        std_visit_y=('mean_fixation_y', 'std'), 
        event_participants=('Event Participants', 'first'), 
        number_of_linked_visits=('Linked Visits Count', 'first')
    ).reset_index()

    # Combine the results
    event_metrics = pd.merge(standard_metrics, custom_metrics.reset_index(), on=['Event ID', 'AOI', 'Group'])
    event_metrics.to_csv('event_metrics.csv')

# Sort the DataFrame by AOI and start_timestamp
final_df = final_df.sort_values(by=['Event ID', 'Group', 'start_timestamp']).reset_index(drop=True)

# Initialize new columns
final_df['Initiator of event'] = 0
final_df['Gaze Follower of event'] = 0

# Iterate through each AOI
for aoi, group in final_df.groupby('AOI'):
    current_initiator = None
    for idx, row in group.iterrows():
        if row['Joint Attention'] == 1:
            if current_initiator is None:
                current_initiator = row['Participant ID']
            if row['Participant ID'] == current_initiator:
                final_df.loc[idx, 'Initiator of event'] = 1
                final_df.loc[idx, 'Gaze Follower of event'] = 0
            else:
                final_df.loc[idx, 'Initiator of event'] = 0
                final_df.loc[idx, 'Gaze Follower of event'] = 1
        else:
            current_initiator = None
            final_df.loc[idx, ['Initiator of event', 'Gaze Follower of event']] = 0

# Ensure that if Joint Attention is false, both Gaze Follower and Initiation are 0
final_df.loc[final_df['Joint Attention'] == 0, ['Initiator of event', 'Gaze Follower of event']] = 0

# Convert 'Joint Attention' to 1 where True, and 0 where False
final_df['Joint Attention'] = final_df['Joint Attention'].astype(int)

final_df['unique_visit_id'] = final_df['AOI'].astype(str) + "_" + final_df['visit_id'].astype(str)

final_df = final_df.drop(columns=['Initiation', 'Gaze Follower']) 

file_path_saving = r'.csv'
final_df.to_csv(file_path_saving, index=False)


