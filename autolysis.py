# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "requests",
#   "scikit-learn",
#   "chardet"
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(
    filename='autolysis.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_api_details():
    """
    Retrieves the AI Proxy token and sets the API endpoint URL.
    """
    api_proxy_token = os.getenv("AIPROXY_TOKEN")
    if not api_proxy_token:
        logging.error("AIPROXY_TOKEN environment variable not set.")
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)
    
    api_proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    return api_proxy_token, api_proxy_url

def detect_encoding(file_path):
    """
    Detects the encoding of a file using chardet.
    """
    try:
        import chardet
    except ImportError:
        logging.info("chardet library not found. Installing...")
        os.system("pip install chardet")
        import chardet
    
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def analyze_dataset(file_path):
    """
    Loads the dataset with appropriate encoding and performs basic and advanced analysis.
    Returns the DataFrame and a dictionary containing analysis details.
    """
    # Detect encoding
    encoding = detect_encoding(file_path)
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        logging.info(f"Successfully loaded {file_path} with encoding {encoding}")
        print(f"Successfully loaded {file_path} with encoding {encoding}")
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)
    
    # Basic Analysis
    try:
        summary_stats = df.describe(include='all').to_dict()
        logging.info(f"Generated summary statistics for {file_path}")
    except Exception as e:
        logging.warning(f"Unable to generate summary statistics for {file_path}. {e}")
        print(f"Warning: Unable to generate summary statistics for {file_path}. {e}")
        summary_stats = {}
    
    missing_values = df.isnull().sum().to_dict()
    dtypes = df.dtypes.apply(str).to_dict()
    columns = list(df.columns)
    
    analysis = {
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing_values,
        "summary_stats": summary_stats
    }
    
    # Handle Missing Values
    # Impute numeric columns with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    
    # Impute categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        mode = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode)
    
    # Advanced Analysis: Outlier Detection
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outliers[col] = outlier_count
    
    analysis["outliers"] = outliers
    
    # Clustering
    if len(numeric_cols) >= 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        df['Cluster'] = clusters
        cluster_counts = df['Cluster'].value_counts().to_dict()
        analysis["clusters"] = cluster_counts
        logging.info(f"Performed K-Means clustering on {file_path}")
    else:
        analysis["clusters"] = "Not enough numeric columns for clustering."
        logging.warning(f"Not enough numeric columns for clustering in {file_path}")
    
    return df, analysis

def generate_visualizations(df, output_dir):
    """
    Generates visualizations based on the DataFrame and saves them as PNG files.
    Returns a list of generated PNG filenames.
    """
    png_files = []
    
    # 1. Correlation Heatmap (if applicable)
    numeric_columns = df.select_dtypes(include='number').columns
    if len(numeric_columns) > 1:
        corr = df[numeric_columns].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        png_files.append("correlation_heatmap.png")
        logging.info(f"Saved correlation_heatmap.png in {output_dir}")
        print(f"Saved correlation_heatmap.png in {output_dir}")
    
    # 2. Distribution Plot of the First Numeric Column
    if len(numeric_columns) > 0:
        first_numeric = numeric_columns[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(df[first_numeric], kde=True, bins=30, color='skyblue')
        plt.title(f"Distribution of {first_numeric}")
        plt.xlabel(first_numeric)
        plt.ylabel("Frequency")
        dist_path = os.path.join(output_dir, f"{first_numeric}_distribution.png")
        plt.savefig(dist_path)
        plt.close()
        png_files.append(f"{first_numeric}_distribution.png")
        logging.info(f"Saved {first_numeric}_distribution.png in {output_dir}")
        print(f"Saved {first_numeric}_distribution.png in {output_dir}")
    
    # 3. Categorical Count Plot (if applicable)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        first_categorical = categorical_columns[0]
        plt.figure(figsize=(12, 8))
        sns.countplot(
            data=df,
            y=first_categorical,
            order=df[first_categorical].value_counts().index[:10],
            palette="viridis",
            hue=first_categorical,
            dodge=False
        )
        plt.title(f"Top 10 {first_categorical} Categories")
        plt.xlabel("Count")
        plt.ylabel(first_categorical)
        plt.legend([], [], frameon=False)  # Hide legend to fix FutureWarning
        count_path = os.path.join(output_dir, f"{first_categorical}_count.png")
        plt.savefig(count_path)
        plt.close()
        png_files.append(f"{first_categorical}_count.png")
        logging.info(f"Saved {first_categorical}_count.png in {output_dir}")
        print(f"Saved {first_categorical}_count.png in {output_dir}")
    
    return png_files

def narrate_story(analysis, png_files, api_proxy_token, api_proxy_url):
    """
    Generates a narrative in Markdown format using the LLM based on the analysis.
    Returns the narrative as a string.
    """
    # Create a concise summary to send to the LLM
    analysis_summary = (
        f"**Columns:** {analysis['columns']}\n"
        f"**Data Types:** {analysis['dtypes']}\n"
        f"**Missing Values:** {analysis['missing_values']}\n"
        f"**Summary Statistics:** {list(analysis['summary_stats'].keys())}\n"
        f"**Outliers Detected:** {analysis['outliers']}\n"
        f"**Clustering Results:** {analysis['clusters']}\n"
    )
    
    # Define the prompt for the LLM
    prompt = (
    "You are an advanced data scientist with expertise in data analysis and visualization. Based on the comprehensive analysis provided below, generate a detailed narrative in Markdown format that includes:\n"
    "1. **Dataset Overview:** A thorough description of the dataset, its source, and its structure.\n"
    "2. **Key Insights:** Highlight the most significant findings and trends observed in the data.\n"
    "3. **Visualization Interpretations:** Provide in-depth explanations of each generated chart, discussing what they reveal about the data.\n"
    "4. **Implications and Recommendations:** Discuss the potential implications of the findings and offer actionable recommendations.\n"
    "5. **Dynamic Analysis Suggestions:** Propose three additional analyses or visualizations that could further enhance the understanding of the dataset.\n"
    "6. **Vision Agentic Enhancements:** Suggest ways to integrate visual analysis or image-based insights to complement the current findings.\n\n"
    f"**Comprehensive Analysis:**\n{analysis_summary}"
)


    
    # Prepare the payload for the AI Proxy
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data scientist narrating the story of a dataset."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_proxy_token}"
    }
    
    try:
        response = requests.post(api_proxy_url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            story = result['choices'][0]['message']['content']
            logging.info("Successfully generated narrative with LLM.")
            print("Successfully generated narrative with LLM.")
        else:
            logging.error(f"Error: {response.status_code}, {response.text}")
            print(f"Error: {response.status_code}, {response.text}")
            story = f"Error generating narrative: {response.status_code}, {response.text}"
    except Exception as e:
        logging.error(f"Error generating narrative: {e}")
        print(f"Error generating narrative: {e}")
        story = f"Error generating narrative: {e}"
    
    # Append image references to the narrative
    if png_files and "error" not in story.lower():
        story += "\n\n## Visualizations\n"
        for img in png_files:
            story += f"![{img}]({img})\n"
    
    # Suggest Additional Analyses
    suggestion_prompt = (
    "Based on the narrative and analysis provided below, suggest three innovative analyses or visualizations that could offer deeper insights or uncover hidden patterns in the dataset. Additionally, recommend how visual (image-based) analysis techniques could be integrated to enhance the overall understanding.\n\n"
    f"**Narrative and Analysis:**\n{story}"
)

    
    suggestion_payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data scientist."},
            {"role": "user", "content": suggestion_prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        suggestion_response = requests.post(api_proxy_url, headers=headers, json=suggestion_payload)
        if suggestion_response.status_code == 200:
            suggestion_result = suggestion_response.json()
            suggestions = suggestion_result['choices'][0]['message']['content']
            logging.info("Successfully received additional analysis suggestions.")
            print("Successfully received additional analysis suggestions.")
        else:
            logging.error(f"Error: {suggestion_response.status_code}, {suggestion_response.text}")
            suggestions = f"Error receiving suggestions: {suggestion_response.status_code}, {suggestion_response.text}"
    except Exception as e:
        logging.error(f"Error receiving suggestions: {e}")
        print(f"Error receiving suggestions: {e}")
        suggestions = f"Error receiving suggestions: {e}"
    
    story += f"\n\n## Additional Suggestions\n{suggestions}"
    
    return story

def analyze_and_generate_output(file_path, api_proxy_token, api_proxy_url):
    """
    Processes a single CSV file: analyzes data, generates visualizations, narrates the story.
    Saves outputs in a dedicated directory.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(".", base_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created directory: {output_dir}")
    print(f"Created directory: {output_dir}")
    
    df, analysis = analyze_dataset(file_path)
    png_files = generate_visualizations(df, output_dir)
    story = narrate_story(analysis, png_files, api_proxy_token, api_proxy_url)
    
    # Write story to README.md
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding='utf-8') as f:
        f.write(story)
    logging.info(f"Saved README.md in {output_dir}")
    print(f"Saved README.md in {output_dir}")
    
    return output_dir

def main():
    """
    Main function to process all provided CSV files.
    """
    if len(sys.argv) < 2:
        print("Usage: uv run neAuto.py data/goodreads.csv data/happiness.csv data/media.csv")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    api_proxy_token, api_proxy_url = get_api_details()
    output_dirs = []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"Processing file: {file_path}")
            logging.info(f"Processing file: {file_path}")
            output_dir = analyze_and_generate_output(file_path, api_proxy_token, api_proxy_url)
            output_dirs.append(output_dir)
        else:
            logging.error(f"File {file_path} not found!")
            print(f"File {file_path} not found!")
    
    print(f"Analysis completed. Results saved in directories: {', '.join(output_dirs)}")
    logging.info(f"Analysis completed. Results saved in directories: {', '.join(output_dirs)}")

if __name__ == "__main__":
    main()
