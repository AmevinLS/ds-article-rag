# ds-article-rag

## Dataset used
- **1300+ Towards DataScience Medium Articles Dataset**
(https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset)


## How to run
### Running with Docker
1. Run `git clone https://github.com/AmevinLS/ds-article-rag`
2. Change working directory to the cloned repository (`cd ds-article-rag`)
3. Download [dataset](https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset) and extract it to `./data/medium.csv` (create the `./data` directory if needed)
4. Run `docker-compose up`
5. Go to `http://localhost:8501` in your browser to open the streamlit app

### Running locally
Make sure you have Ollama installed (you can download it here [here](https://ollama.com/)). \
*Optionally, you can run* `ollama pull llama2` *yourself to avoid download issues at runtime*

1. Run `git clone https://github.com/AmevinLS/ds-article-rag`
2. Change working directory to the cloned repository (`cd ds-article-rag`)
3. Download [dataset](https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset) and extract it to `./data/medium.csv` (create the `./data` directory if needed)
4. Run `pip install -r requirements.txt` (tested for Python=3.9)
5. Run `streamlit run ds_article_rag/app.py`
5. Go to `http://localhost:8501` in your browser to open the streamlit app

### [Optional] Download cache for faster start-up
1. Download 'cache' folder from Google Drive ([here](https://drive.google.com/drive/folders/1zCkBSJxQ0T_nCzr4UxEuU4_wQmHZ8pbj?usp=sharing))
2. Export it into the `./data`, so the resulting structure looks as follows:
```
data/
  medium.csv
  cache/
    code_reduced_par_embeds.npy
    code_reduced_paragraphs.csv
```