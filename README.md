# Quora Question Answering Model
## Project Overview
This project aims to develop a state-of-the-art question-answering model using the Quora Question Answer Dataset. The objective is to analyze the dataset, apply NLP techniques, test various models, and evaluate their performance using standard metrics. The final goal is to present the findings through comprehensive visualizations and a detailed report.

## Tech Stack
### Frontend
* #### Plotly:
  Used for creating interactive visualizations to analyze model performance.
### Backend
* #### Python:
  The main programming language used for data processing, model training, and evaluation.
* #### Hugging Face Transformers:
  Utilized for loading and running BERT, T5, and GPT models.
* ##### Pandas:
  Used for data manipulation and analysis.
* #### NLTK:
  Used for tokenization and evaluation metrics (BLEU).
* #### Rouge Score:
  Used for computing ROUGE scores.
* #### Matplotlib:
  Used for static data visualizations.
* #### Seaborn:
  Used for enhanced statistical visualizations.
### Additional Libraries
* #### scikit-learn:
  Used for computing F1 scores.
* #### NumPy:
  Utilized for numerical operations.
* #### Plotly:
  Used for creating interactive and detailed visualizations.
### Project Workflow
1. #### Data Loading and Cleaning:
   * Load the Quora Question Answer Dataset.
   * Select relevant columns (questions and answers) and remove irrelevant information.
    
2. #### Preprocessing:
   * Tokenize the questions using Hugging Face tokenizers.
  
4. #### Model Loading and Inference:
   * Load BERT, T5, and GPT models.
   * Generate outputs for each model based on the tokenized questions.

4. #### Evaluation Metrics:
   * Compute BLEU, ROUGE, and F1 scores for each model's outputs to evaluate performance.

6. #### Visualizations:
   * Create visualizations using Matplotlib, Seaborn, and Plotly to show:
     * Data distribution (question and answer length).
     * Feature importance.
     * Model performance (BLEU, ROUGE, and F1 scores).
### How to Run
1. #### Install Dependencies:
   ```python
   pip install pandas nltk transformers rouge-score scikit-learn matplotlib seaborn plotly
   ```

3. #### Run the Script:
   * Execute the combined code script provided in this project to load the dataset, preprocess the data, generate model outputs, compute evaluation metrics, and create visualizations.
### Results and Findings
  The results of the project include:
  * Detailed analysis of the dataset distribution.
  * Comparison of model performance across BERT, T5, and GPT models using BLEU, ROUGE, and F1 scores.
  * Interactive and static visualizations that provide insights into the data and model evaluations.
The findings and visualizations will help in understanding the strengths and weaknesses of each model and provide a foundation for further improvements and research in question-answering systems.

### Conclusion
This project demonstrates the process of developing and evaluating a question-answering model using state-of-the-art NLP techniques and models. The comprehensive analysis and visualizations offer valuable insights and pave the way for future enhancements.

For more details, refer to the provided code and visualizations.
