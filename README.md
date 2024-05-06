# ds-article-rag

## Dataset used
- **1300+ Towards DataScience Medium Articles Dataset**
(https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset)


## How to run
**Note**: Startup of the app as well as LLM querying can take a long time, *especially* without a GPU. \
If you don't have llama2 downloaded in ollama (or your running with docker), first invocation of LLM querying can be very time-consuming due to ollama downloading the llama2 model (it is known to have connection issues).

*[16GB RAM, Intel Core i7-9750H CPU, Nvidia GeForce 1660-Ti] -- 3 minutes startup with GPU (no cache) -- 10 minutes startup CPU-only (no cache)*

### Running with Docker
1. Run `git clone https://github.com/AmevinLS/ds-article-rag`
2. Change working directory to the cloned repository (`cd ds-article-rag`)
3. Download [dataset](https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset) and extract it to `./data/medium.csv` (create the `./data` directory if needed)
4. Run:
    - `docker-compose -f docker-compose_cpu.yml up --build` to run on CPU, if you don't have an Nvidia GPU
    - `docker-compose -f docker-compose_gpu.yml up --build` to run using your Nvidia GPU
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

## Demo
![](./docs/demonstration.gif)

## Details on the system
You can find the details pertaining to how the whole system is structured and other relevant information in the `./docs/report.md` file in this repository
