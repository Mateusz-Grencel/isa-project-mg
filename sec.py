import pandas as pd
import re
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
from nomic import AtlasDataset

# Initialize Colorama
init(autoreset=True)

def main_program():
    while True:
        # Get dataset ID from user with error handling
        while True:
            try:
                dataset_id = input("Enter the dataset ID from Nomic: ")

                # Initialize AtlasDataset with dynamically provided identifier
                dataset = AtlasDataset(identifier=dataset_id)

                # Retrieve the first map from the dataset
                map = dataset.maps[0]
                print(map)

                print(f"Successfully loaded dataset with ID: {dataset_id}")
                print(f"Loading analyses...")
                break  # Exit loop if everything went well
            except Exception as e:
                print(f"Error: {e}. Please try again.")

        while True:
            # Ask for knowledge graph ID
            id_input = input("Enter the knowledge graph ID: ")

            # Get neighbors for the specified point
            neighbor, distances = map.embeddings.vector_search(ids=[id_input], k=30)

            # Sample data analysis
            data = []
            for neighbor_id in neighbor[0]:
                neighbor_info = dataset.get_data(ids=[neighbor_id])
                if neighbor_info:
                    data.append({
                        'id': neighbor_info[0]['id_'],
                        'title': neighbor_info[0]['title'],
                        'link': neighbor_info[0]['link'],
                        'meta_description': neighbor_info[0]['meta_description']
                    })

            df = pd.DataFrame(data)

            # Main menu for selecting analysis
            while True:
                print("\nSelect analysis to perform:")
                print("1. Meta Description Length Analysis")
                print("2. Keyword in Meta Description Analysis")
                print("3. Keyword in Title Analysis")
                print("4. Title Length Analysis")
                print("5. Duplicate Title Analysis")
                print("6. URL Length Analysis")
                print("7. Parameters in URL Analysis")
                print("8. Enter new knowledge graph ID")
                print("9. Exit program")

                choice = input("Enter your choice (1-9): ")

                if choice == '1':
                    meta_desc_length_analysis(df, id_input)
                elif choice == '2':
                    keyword_in_meta_description_analysis(df, id_input)
                elif choice == '3':
                    keyword_in_title_analysis(df, id_input)
                elif choice == '4':
                    title_length_analysis(df, id_input)
                elif choice == '5':
                    duplicate_title_analysis(df, id_input)
                elif choice == '6':
                    url_length_analysis(df, id_input)
                elif choice == '7':
                    params_in_url_analysis(df, id_input)
                elif choice == '8':
                    # Exit the current loop and go back to dataset selection
                    print("Returning to dataset selection.")
                    break
                elif choice == '9':
                    print("Exiting the program.")
                    return  # Exit the entire program
                else:
                    print("Invalid choice. Please enter a number between 1 and 9.")

# Functions for data analysis with plots
def meta_desc_length_analysis(df, id_input):
    # Get user input for minimum and maximum meta description length with error handling
    while True:
        try:
            min_length = int(input("Enter the minimum meta description length: "))
            max_length = int(input("Enter the maximum meta description length: "))
            if min_length < 0 or max_length < 0 or min_length > max_length:
                print("Please enter valid lengths (non-negative and min should be less than or equal to max).")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter numeric values for lengths.")

    # Calculate meta description lengths
    df["meta_desc_length"] = df["meta_description"].str.len()

    # Filter the DataFrame to include only meta descriptions within the length range
    filtered_df = df[(df["meta_desc_length"] >= min_length) & (df["meta_desc_length"] <= max_length)]

    print(f"Meta descriptions with length between {min_length} and {max_length} characters:")

    # Highlight rows in the filtered DataFrame
    highlight_rows_local(id_input, filtered_df, ['id', 'meta_desc_length', 'meta_description'])

    # Create a list for y-values where each value corresponds to whether there's at least one entry for that length
    y_values = [1 if count > 0 else 0 for count in range(min_length, max_length + 1)]

    # Create a histogram with bars for each length, using y_values
    plt.bar(range(min_length, max_length + 1), y_values, color='green', edgecolor='black')

    plt.title(f"Meta Description Lengths Between {min_length} and {max_length}")
    plt.xlabel("Meta Description Length (character count)")
    plt.ylabel("Presence of Entries (0 or 1)")

    # Set y-axis ticks to 0 and 1
    plt.yticks([0, 1])

    # Set x-axis ticks from min_length to max_length with a step of 2
    plt.xticks(range(min_length, max_length + 1, 2))

    plt.tight_layout()
    plt.show()

    input("Press Enter to return to the menu...")



# Other functions remain the same, just add 'df' and 'id_input' as arguments.

def keyword_in_meta_description_analysis(df, id_input):
    keyword = input("Enter the keyword to search in the meta description: ")
    keyword_pattern = re.escape(keyword)

    # Count occurrences of the keyword in meta descriptions
    df['keyword_count'] = df['meta_description'].str.count(rf'\b{keyword_pattern}\b', flags=re.IGNORECASE)

    print(f"Number of occurrences of the keyword '{keyword}' in the meta description:")

    # Highlight rows with keyword count
    highlight_rows_local(id_input, df, ['id', 'keyword_count', 'meta_description'])

    # Separate counts for 0 and 1+ occurrences
    count_zero = (df['keyword_count'] == 0).sum()
    count_one_or_more = (df['keyword_count'] >= 1).sum()

    # Create the bar plot manually with custom colors
    plt.bar([0, 1], [count_zero, count_one_or_more], color=['red', 'green'], width=0.6, edgecolor='black')

    # Set plot titles and labels
    plt.title(f"Occurrences of the Keyword '{keyword}' in the Meta Description")
    plt.xlabel("Number of Occurrences (0 or 1+)")
    plt.ylabel("Number of Entries")

    # Set the y-axis to have values from 0 to 30
    plt.ylim(0, 30)
    plt.yticks(range(0, 31, 1))  # y-axis intervals from 0 to 30

    # Force x-axis ticks to show only 0 and 1
    plt.xticks([0, 1], labels=["0 occurrences", "1+ occurrences"])

    # Show the plot
    plt.tight_layout()
    plt.show()

    input("Press Enter to return to the menu...")



def keyword_in_title_analysis(df, id_input):
    keyword = input("Enter the keyword to search in the title: ")
    keyword_pattern = re.escape(keyword)

    # Count occurrences of the keyword in titles
    df['keyword_count'] = df['title'].str.count(rf'\b{keyword_pattern}\b', flags=re.IGNORECASE)

    print(f"Number of occurrences of the keyword '{keyword}' in the title:")

    # Highlight rows with keyword count
    highlight_rows_local(id_input, df, ['id', 'keyword_count', 'title'])

    # Separate counts for 0 and 1+ occurrences
    count_zero = (df['keyword_count'] == 0).sum()
    count_one_or_more = (df['keyword_count'] >= 1).sum()

    # Create the bar plot manually with custom colors
    plt.bar([0, 1], [count_zero, count_one_or_more], color=['red', 'green'], width=0.6, edgecolor='black')

    # Set plot titles and labels
    plt.title(f"Occurrences of the Keyword '{keyword}' in the Title")
    plt.xlabel("Number of Occurrences (0 or 1+)")
    plt.ylabel("Number of Titles")

    # Set the y-axis to have values from 0 to 30
    plt.ylim(0, 30)
    plt.yticks(range(0, 31, 1))  # y-axis intervals from 0 to 30

    # Force x-axis ticks to show only 0 and 1
    plt.xticks([0, 1], labels=["0 occurrences", "1+ occurrences"])

    # Show the plot
    plt.tight_layout()
    plt.show()

    input("Press Enter to return to the menu...")


def title_length_analysis(df, id_input):
    # Get user input for minimum and maximum title length with error handling
    while True:
        try:
            min_length = int(input("Enter the minimum title length: "))
            max_length = int(input("Enter the maximum title length: "))
            if min_length < 0 or max_length < 0 or min_length > max_length:
                print("Please enter valid lengths (non-negative and min should be less than or equal to max).")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter numeric values for lengths.")

    # Calculate the title lengths
    df['title_len'] = df['title'].str.len()

    # Filter the DataFrame based on the specified title lengths
    filtered_df = df[(df["title_len"] >= min_length) & (df["title_len"] <= max_length)]

    print(f"Titles with length between {min_length} and {max_length} characters:")

    # Highlight rows with the filtered data
    highlight_rows_local(id_input, filtered_df, ['id', 'title_len', 'title'])

    # Count occurrences of each title length in the filtered DataFrame
    counts = filtered_df['title_len'].value_counts().sort_index()

    # Create a histogram with two bins: one for 0 occurrences and another for 1 occurrence
    plt.bar(counts.index, counts.values, color=['red' if x == 0 else 'green' for x in counts.values], edgecolor='black')

    plt.title(f"Title Lengths ({min_length}-{max_length} characters)")
    plt.xlabel("Title Length (character count)")
    plt.ylabel("Number of Titles")

    # Set y-axis ticks to 0 and 1
    plt.yticks([0, 1])

    # Set x-axis ticks from min_length to max_length with a step of 2
    plt.xticks(range(min_length, max_length + 1, 2))

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

    input("Press Enter to return to the menu...")


def duplicate_title_analysis(df, id_input):
    # Find duplicate titles
    duplicate_titles = df[df.duplicated(subset='title', keep=False)]

    print("Duplicate Titles:")
    highlight_rows_local(id_input, duplicate_titles, ['id', 'title', 'link', 'meta_description'])

    # Count occurrences of titles and find duplicates
    title_counts = df['title'].value_counts()
    duplicate_count = title_counts[title_counts > 1]

    # Plotting the number of duplicate titles
    plt.hist(duplicate_count, bins=10, color='red', edgecolor='black')
    plt.title("Number of Duplicate Titles")
    plt.xlabel("Number of Duplicates")
    plt.ylabel("Number of Titles")
    plt.tight_layout()
    plt.show()

    input("Press Enter to return to the menu...")

def url_length_analysis(df, id_input):
    # Get user input for minimum and maximum URL length with error handling
    while True:
        try:
            min_length = int(input("Enter the minimum URL length: "))
            max_length = int(input("Enter the maximum URL length: "))
            if min_length < 0 or max_length < 0 or min_length > max_length:
                print("Please enter valid lengths (non-negative and min should be less than or equal to max).")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter numeric values for lengths.")

    # Calculate URL lengths
    df['url_length'] = df['link'].apply(len)

    # Filter the DataFrame to include only URLs within the specified length range
    filtered_df = df[(df['url_length'] >= min_length) & (df['url_length'] <= max_length)]

    print("Filtered URL Lengths:")

    highlight_rows_local(id_input, filtered_df, ['id', 'url_length', 'link'])

    # Plotting URL lengths
    plt.hist(filtered_df['url_length'], bins=range(min_length, max_length + 2), color='cyan', edgecolor='black')
    plt.title("URL Lengths")
    plt.xlabel("URL Length (character count)")
    plt.ylabel("Number of URLs")

    # Set x-axis ticks from min_length to max_length with a step of 5
    plt.xticks(range(min_length, max_length + 1, 5))

    # Set y-axis ticks to show whole numbers
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

    input("Press Enter to return to the menu...")


def params_in_url_analysis(df, id_input):
    # Check if the URL has parameters
    df['has_params'] = df['link'].apply(lambda x: '?' in x)

    # Count the number of parameters in each URL
    df['params_count'] = df['link'].apply(lambda x: len(x.split('?')[1].split('&')) if '?' in x else 0)

    # Extract query parameters for display
    df['query_params'] = df['link'].apply(lambda x: x.split('?')[1] if '?' in x else '')

    print("URLs with Parameters:")
    for index, row in df.iterrows():
        if row['has_params']:
            print(f"ID: {row['id']}, URL: {row['link']}, Query Parameter: {row['query_params']}")

    # Highlight rows in the DataFrame with ID, URL, and parameter count
    highlight_rows_local(id_input, df, ['id', 'link', 'params_count'])

    # Plotting the number of parameters in URLs
    plt.hist(df['params_count'], bins=10, color='cyan', edgecolor='black')
    plt.title("Parameters in URLs")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Number of URLs")

    # Set y-axis ticks to show whole numbers
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Set x-axis ticks to show whole numbers
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

    input("Press Enter to return to the menu...")


def highlight_rows_local(id_value, df, cols_to_show):
    for index, row in df.iterrows():
        output = row[cols_to_show].tolist()
        if str(row['id']) == str(id_value):
            print(Fore.YELLOW + str(output) + Style.RESET_ALL)  # Highlight the row
        else:
            print(str(output))  # Regular display of the row

# Start the main program
main_program()
