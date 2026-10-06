# 2026 College Football Predictor

A Streamlit website wrapper around the validated Colab prediction pipeline.

## First-time setup

### 1. Export the model bundle from Colab
Run the working notebook through Cell 37C.

Then create a new Colab cell and paste the contents of `export_bundle.py`.
Run it. Colab downloads:

`model_bundle.pkl`

### 2. Put the bundle in this project folder
The folder should contain:

- `app.py`
- `model_bundle.pkl`
- `requirements.txt`
- `.gitignore`

### 3. Test locally (optional)
From a terminal in this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 4. Regression checks
Before publishing, verify the website reproduces the working notebook using the same exported model state:

- Texas A&M @ Texas: approximately Texas by 9.75
- Indiana @ Nebraska: approximately Indiana by 0.51

If either differs materially, stop and debug before deployment.

### 5. GitHub / Streamlit
Create a GitHub repository and upload the project files. Then connect that repository to Streamlit Community Cloud and deploy `app.py`.

## Updating the model later

When the Colab model changes:

1. Run the updated notebook through the final working cells.
2. Run `export_bundle.py` again.
3. Replace `model_bundle.pkl` in the website project.
4. Re-run regression tests.
5. Push the approved update to GitHub.

This keeps the public interface separate from model development.
