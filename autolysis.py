# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "requests",
#   "scikit-learn",
#   "chardet",
#   "plotly",
#   "numpy"
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import numpy as np
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

def determine_cluster_count(df):
    """
    Determines the optimal number of clusters using the silhouette score.
    """
    numeric_data = df.select_dtypes(include=['number'])
    if numeric_data.empty:
        return 1  # Default to 1 cluster if no numeric data
    
    max_clusters = min(10, len(numeric_data) // 10)  # Prevent too many clusters
    if max_clusters < 2:
        return 1
    
    best_score = -1
    best_k = 2
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(numeric_data)
        try:
            score = silhouette_score(numeric_data, clusters)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    return best_k

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
    
    # Feature Importance using Random Forest (if target variable exists)
    feature_importance = {}
    target_col = None
    # Attempt to identify a target column (binary or categorical)
    for col in categorical_cols:
        if df[col].nunique() <= 10:  # Simple heuristic
            target_col = col
            break
    if target_col and len(numeric_cols) >= 1:
        try:
            X = df.select_dtypes(include=['number']).drop(columns=[target_col], errors='ignore')
            y = df[target_col]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            feature_importance = dict(zip(X.columns, importances))
            analysis["feature_importance"] = feature_importance
            logging.info(f"Computed feature importance using Random Forest for {file_path}")
        except Exception as e:
            logging.warning(f"Unable to compute feature importance for {file_path}. {e}")
            analysis["feature_importance"] = {}
    else:
        analysis["feature_importance"] = {}
    
    # Principal Component Analysis (PCA)
    pca = None
    if len(numeric_cols) >= 2:
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(scaled_data)
            df['PC1'] = principal_components[:, 0]
            df['PC2'] = principal_components[:, 1]
            analysis["pca"] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
            }
            logging.info(f"Performed PCA on {file_path}")
        except Exception as e:
            logging.warning(f"Unable to perform PCA for {file_path}. {e}")
            analysis["pca"] = {}
    else:
        analysis["pca"] = {}
    
    # Clustering
    if len(numeric_cols) >= 2:
        optimal_k = determine_cluster_count(df)
        if optimal_k > 1:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            df['Cluster'] = clusters
            cluster_counts = df['Cluster'].value_counts().to_dict()
            analysis["clusters"] = cluster_counts
            logging.info(f"Performed K-Means clustering with k={optimal_k} on {file_path}")
        else:
            analysis["clusters"] = "Clustering not performed due to insufficient data or low optimal k."
            logging.warning(f"Clustering not performed for {file_path}")
    else:
        analysis["clusters"] = "Not enough numeric columns for clustering."
        logging.warning(f"Not enough numeric columns for clustering in {file_path}")
    
    return df, analysis

def generate_visualizations(df, output_dir):
    """
    Generates visualizations based on the DataFrame and saves them as PNG/HTML files.
    Returns a list of generated filenames.
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
    
    # 4. Box Plot for Outlier Detection
    if len(numeric_columns) > 0:
        first_numeric = numeric_columns[0]
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[first_numeric], color='lightgreen')
        plt.title(f"Box Plot of {first_numeric}")
        plt.xlabel(first_numeric)
        box_path = os.path.join(output_dir, f"{first_numeric}_boxplot.png")
        plt.savefig(box_path)
        plt.close()
        png_files.append(f"{first_numeric}_boxplot.png")
        logging.info(f"Saved {first_numeric}_boxplot.png in {output_dir}")
        print(f"Saved {first_numeric}_boxplot.png in {output_dir}")
    
    # 5. Scatter Plot for Clustering (if applicable)
    if 'Cluster' in df.columns and len(numeric_columns) >= 2:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x=numeric_columns[0],
            y=numeric_columns[1],
            hue='Cluster',
            palette='Set1'
        )
        plt.title(f"Scatter Plot of {numeric_columns[0]} vs {numeric_columns[1]} with Clusters")
        scatter_path = os.path.join(output_dir, f"{numeric_columns[0]}_vs_{numeric_columns[1]}_clusters.png")
        plt.savefig(scatter_path)
        plt.close()
        png_files.append(f"{numeric_columns[0]}_vs_{numeric_columns[1]}_clusters.png")
        logging.info(f"Saved {numeric_columns[0]}_vs_{numeric_columns[1]}_clusters.png in {output_dir}")
        print(f"Saved {numeric_columns[0]}_vs_{numeric_columns[1]}_clusters.png in {output_dir}")
    
    # 6. Interactive Plotly Visualization (Optional)
    if len(numeric_columns) >= 2:
        try:
            fig = px.scatter(
                df,
                x=numeric_columns[0],
                y=numeric_columns[1],
                color='Cluster' if 'Cluster' in df.columns else None,
                title=f"Interactive Scatter Plot of {numeric_columns[0]} vs {numeric_columns[1]}"
            )
            interactive_plot_path = os.path.join(output_dir, f"{numeric_columns[0]}_vs_{numeric_columns[1]}_interactive.html")
            fig.write_html(interactive_plot_path)
            png_files.append(f"{numeric_columns[0]}_vs_{numeric_columns[1]}_interactive.html")
            logging.info(f"Saved {numeric_columns[0]}_vs_{numeric_columns[1]}_interactive.html in {output_dir}")
            print(f"Saved {numeric_columns[0]}_vs_{numeric_columns[1]}_interactive.html in {output_dir}")
        except Exception as e:
            logging.warning(f"Unable to create interactive Plotly visualization for {file_path}. {e}")
    
    # 7. PCA Plot (if applicable)
    if 'PC1' in df.columns and 'PC2' in df.columns:
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=df,
                x='PC1',
                y='PC2',
                hue='Cluster' if 'Cluster' in df.columns else None,
                palette='Set2'
            )
            plt.title("PCA Scatter Plot")
            pca_path = os.path.join(output_dir, "pca_scatter_plot.png")
            plt.savefig(pca_path)
            plt.close()
            png_files.append("pca_scatter_plot.png")
            logging.info(f"Saved pca_scatter_plot.png in {output_dir}")
            print(f"Saved pca_scatter_plot.png in {output_dir}")
        except Exception as e:
            logging.warning(f"Unable to create PCA scatter plot for {file_path}. {e}")
    
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
        f"**Feature Importance:** {analysis.get('feature_importance', {})}\n"
        f"**Clustering Results:** {analysis['clusters']}\n"
        f"**PCA Explained Variance:** {analysis.get('pca', {}).get('explained_variance_ratio', [])}\n"
    )
    
    # Enhanced Narrative Generation Prompt
    prompt = (
        "You are an expert data scientist with extensive experience in data analysis and visualization. Based on the comprehensive analysis provided below, generate a detailed narrative in Markdown format that includes the following sections:\n"
        "1. **Dataset Overview:** A thorough description of the dataset, including its source, purpose, and structure.\n"
        "2. **Data Cleaning and Preprocessing:** Outline the steps taken to handle missing values, outliers, and any data transformations applied.\n"
        "3. **Exploratory Data Analysis (EDA):** Present key insights, trends, and patterns discovered during the analysis.\n"
        "4. **Visualizations:** For each generated chart, provide an in-depth explanation of what it represents and the insights it offers.\n"
        "5. **Feature Importance:** Discuss the importance of different features based on the analysis.\n"
        "6. **Clustering and Segmentation:** Discuss the results of any clustering algorithms used, including the characteristics of each cluster.\n"
        "7. **Principal Component Analysis (PCA):** Explain the PCA results and how they contribute to understanding the dataset.\n"
        "8. **Implications and Recommendations:** Based on the findings, suggest actionable recommendations or potential implications for stakeholders.\n"
        "9. **Future Work:** Propose three additional analyses or visualizations that could further enhance the understanding of the dataset.\n"
        "10. **Vision Agentic Enhancements:** Recommend ways to incorporate advanced visual (image-based) analysis techniques or interactive visualizations to provide deeper insights.\n\n"
        f"**Comprehensive Analysis:**\n{analysis_summary}"
    )
    
    # Prepare the payload for the AI Proxy
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data scientist narrating the story of a dataset."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2500,
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
            if img.endswith('.html'):
                story += f"[Interactive Visualization]({img})\n"
            else:
                story += f"![{img}]({img})\n"
    
    # Refined Additional Suggestions Prompt
    suggestion_prompt = (
        "Based on the following narrative and analysis, suggest three innovative analyses or visualizations that could provide deeper insights or uncover hidden patterns in the dataset. Additionally, recommend how advanced visual (image-based) analysis techniques or interactive visualizations could be integrated to enhance the overall understanding.\n\n"
        f"**Narrative and Analysis:**\n{story}"
    )
    
    suggestion_payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data scientist."},
            {"role": "user", "content": suggestion_prompt}
        ],
        "max_tokens": 800,
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
    
    # Vision Agentic Enhancements Prompt
    vision_prompt = (
        "In addition to the existing analyses and visualizations, suggest three interactive visualization techniques or image-based analysis methods that could be integrated into the report to enhance data exploration and stakeholder engagement. Provide brief descriptions of how each technique can be applied to the current dataset.\n"
    )
    
    vision_payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data scientist."},
            {"role": "user", "content": vision_prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }
    
    try:
        vision_response = requests.post(api_proxy_url, headers=headers, json=vision_payload)
        if vision_response.status_code == 200:
            vision_result = vision_response.json()
            vision_suggestions = vision_result['choices'][0]['message']['content']
            logging.info("Successfully received vision agentic enhancements suggestions.")
            print("Successfully received vision agentic enhancements suggestions.")
        else:
            logging.error(f"Error: {vision_response.status_code}, {vision_response.text}")
            vision_suggestions = f"Error receiving suggestions: {vision_response.status_code}, {vision_response.text}"
    except Exception as e:
        logging.error(f"Error receiving vision suggestions: {e}")
        print(f"Error receiving vision suggestions: {e}")
        vision_suggestions = f"Error receiving vision suggestions: {e}"
    
    story += f"\n\n## Vision Agentic Enhancements\n{vision_suggestions}"
    
    return story


def generate_interactive_visualizations(df, output_dir):
    """
    Generates interactive visualizations using Plotly and saves them as HTML files.
    Returns a list of generated HTML filenames.
    """
    interactive_files = []

    numeric_columns = df.select_dtypes(include='number').columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    if len(numeric_columns) >= 2:
        fig = px.scatter(
            df,
            x=numeric_columns[0],
            y=numeric_columns[1],
            color='Cluster' if 'Cluster' in df.columns else None,
            title=f"Interactive Scatter Plot of {numeric_columns[0]} vs {numeric_columns[1]}",
            hover_data=numeric_columns.tolist()
        )
        interactive_plot_path = os.path.join(output_dir, f"{numeric_columns[0]}_vs_{numeric_columns[1]}_interactive.html")
        fig.write_html(interactive_plot_path)
        interactive_files.append(f"{numeric_columns[0]}_vs_{numeric_columns[1]}_interactive.html")
        logging.info(f"Saved {numeric_columns[0]}_vs_{numeric_columns[1]}_interactive.html in {output_dir}")
        print(f"Saved {numeric_columns[0]}_vs_{numeric_columns[1]}_interactive.html in {output_dir}")

    # Additional interactive visualizations can be added here

    return interactive_files




def narrate_story_vision_agentic(analysis, png_files, interactive_files, api_proxy_token, api_proxy_url):
    """
    Generates a narrative including vision agentic enhancements.
    """
    # Existing narrative generation steps...

    # After generating the basic story
    story = narrate_story_dynamic(analysis, png_files, api_proxy_token, api_proxy_url)

    # Analyze visualizations using vision models
    vision_insights = analyze_visualizations(png_files, api_proxy_token, api_proxy_url)

    # Append vision insights to the narrative
    if vision_insights:
        story += "\n\n## Vision Insights\n"
        for img, insights in vision_insights.items():
            story += f"### {img}\n{insights}\n"

    # Include interactive visualizations in the narrative
    if interactive_files:
        story += "\n\n## Interactive Visualizations\n"
        for html in interactive_files:
            story += f"[{html}]({html})\n"

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
    interactive_files = generate_interactive_visualizations(df, output_dir)
    story = narrate_story_vision_agentic(analysis, png_files, interactive_files, api_proxy_token, api_proxy_url)

    # Write story to README.md
    readme_path = os.path.join(output_dir, "README.md")
    try:
        with open(readme_path, "w", encoding='utf-8') as f:
            f.write(story)
        logging.info(f"Saved README.md in {output_dir}")
        print(f"Saved README.md in {output_dir}")
    except Exception as e:
        logging.error(f"Error writing README.md in {output_dir}: {e}")
        print(f"Error writing README.md in {output_dir}: {e}")

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
