# AI-Powered Keyword Clustering Tool

This tool uses NLP (specifically BERT embeddings) to group keywords into semantically similar clusters. It's ideal for SEO professionals who want to organize large lists of keywords for content siloing, avoiding keyword cannibalization, and improving targeting.

## Features

- Uses `Sentence Transformers` for semantic embeddings
- Clusters using DBSCAN (adjustable sensitivity)
- Outputs a CSV with keyword-cluster mapping

## Requirements

- Python 3.8+
- Libraries: `sentence-transformers`, `scikit-learn`, `pandas`

## How to Use

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Place your keywords in `data/sample_keywords.csv`
4. Run the script: `python src/cluster_keywords.py`
5. Output saved in `data/clustered_keywords.csv`

## Sample Output

| Top queries                        | cluster |
|-----------------------------------|---------|
| duo                               | 0       |
| duo mobile                        | 0       |
| duo security                      | 1       |
| duo admin                         | 2       |

## Future Improvements

- Add optional visualization
- Allow user to choose clustering method
- Build a simple UI using Gradio or Streamlit