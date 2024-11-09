# Alzheimer's Disease Prediction

This repository hosts a deep learning model designed to predict the early onset of Alzheimer’s disease using fMRI scans. Leveraging both TensorFlow and PyTorch frameworks, the model analyzes cross-sectional brain images to identify early signs of Alzheimer’s, aiming to assist in early diagnosis and potentially improve patient outcomes.

## Project Overview

Alzheimer’s disease is a progressive brain disorder that affects memory and cognitive function. Early detection can be crucial in managing and potentially slowing the progression of symptoms. This project utilizes deep learning algorithms to analyze fMRI scans, identifying subtle changes in brain structure that may indicate early onset Alzheimer's disease.

## Getting Started

Follow these instructions to set up the project on your local machine for development, testing, and experimentation.

### Prerequisites

To run this project, you will need the following:

- **Python 3.8+**
- **TensorFlow** for deep learning (preferably TensorFlow 2.x)
- **PyTorch** for secondary model development and experimentation
- **NumPy** and **Pandas** for data manipulation
- **Matplotlib** or **Seaborn** for data visualization
- **scikit-learn** for auxiliary machine learning utilities

#### Installation Example

1. Clone the repository:

   ```bash
   git clone https://github.com/devanshalok/alzheimer-s-Disease-prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd alzheimers-prediction
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download fMRI datasets:

   To proceed, you will need an fMRI dataset containing Alzheimer’s and control group images. Please refer to the appropriate open-source datasets (e.g., ADNI) and follow their terms of use.

### Data Preprocessing

Preprocessing includes:

1. **Data Normalization** - Standardizing fMRI data for uniform input to the model.
2. **Segmentation** - Extracting relevant brain cross-sections.
3. **Label Encoding** - Labeling data based on Alzheimer's diagnosis status.

Run the preprocessing script:

```bash
python preprocess_data.py
```

### Model Training

You can train the model using TensorFlow or PyTorch by running:

#### TensorFlow

```bash
python train_model_tensorflow.py
```

#### PyTorch

```bash
python train_model_pytorch.py
```

The training scripts will save model checkpoints and logs for evaluation and visualization.

## Running the Tests

Test scripts are included to validate model accuracy and evaluate the predictive performance.

### End-to-End Tests

End-to-end tests validate the entire pipeline from data input to final Alzheimer’s prediction. Run tests with:

```bash
python test_model.py
```

### Performance Metrics

The model is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
  
Each metric is calculated to ensure the model’s reliability in detecting Alzheimer’s at an early stage.

## Deployment

For deployment, you can save the trained model in the desired format (e.g., `.h5` for TensorFlow or `.pt` for PyTorch) and integrate it into a web-based diagnostic tool or local application.

Example for saving a TensorFlow model:

```python
model.save('alzheimers_model.h5')
```

Example for saving a PyTorch model:

```python
torch.save(model.state_dict(), 'alzheimers_model.pt')
```

## Built With

- **TensorFlow** - Primary deep learning framework
- **PyTorch** - Secondary deep learning framework for model experimentation
- **scikit-learn** - Evaluation metrics and data manipulation
- **NumPy & Pandas** - Data manipulation and processing


## Authors

- **Devansh Alok** - Initial work - [devanshalok](https://github.com/devanshalok)

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Acknowledgments

- **ADNI** - Alzheimer's Disease Neuroimaging Initiative for dataset resources
- Inspiration from various research papers and Alzheimer’s detection studies.
- Gratitude to the community for ongoing research contributions in early Alzheimer's detection.
