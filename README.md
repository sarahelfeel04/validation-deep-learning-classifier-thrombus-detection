# Validation of Deep Learning Classifier for Thrombus Detection

This repository contains code for validating a deep learning-based classifier for detecting thrombi in DSA (Digital Subtraction Angiography) image sequences, specifically for stroke treatment applications.

## Project Structure

- **extraction/**: Contains scripts for extracting DSA sequences from LMDB databases and converting them to NIFTI format.
  - `extractDataFromDb.py`: Extracts all DSA sequences from an LMDB database as 4D NIFTI files.

- **inference/**: Contains the inference pipeline and model definitions.
  - `Classificator.py`: Main classification script.
  - `classifyAllData.py`: Script to classify all data.
  - `CnnLstmModel.py`: CNN-LSTM model architecture.
  - `CustomTransforms.py`: Custom data transformations.
  - `DataAugmentation.py`: Data augmentation utilities.
  - `ImageUtils.py`: Image processing utilities.
  - `IndexTracker.py`: Utility for tracking indices.

- **fine-tuning/**: Scripts for fine-tuning the deep learning model.
  - `fineTuneMixKFold.py`: Performs fine-tuning using mixed k-fold cross-validation.

## Disclaimers

Some of the work in this repository is based on original research by Mittmann et al. (2021) and Baumgartner et al. (2023). Please refer to the following publications for the foundational studies:

- T. Baumgärtner et al., "Towards clinical translation of deep learning-based classification of dsa image sequences for stroke treatment," in Bildverarbeitung für die Medizin 2023, T. M. Deserno et al., Eds. Springer Fachmedien Wiesbaden, 2023, pp. 95–101. doi: 10.1007/978-3-658-41657-7˙22.

- B. J. Mittmann et al., "Deep learning-based classification of dsa image sequences of patients with acute ischemic stroke," International Journal of Computer Assisted Radiology and Surgery, vol. 17, no. 9, pp. 1633–1641, 2022. doi: 10.1007/s11548-022-02654-8.

For full context, complete code, and detailed results, please visit the main repository: [https://github.com/sarahelfeel04/DeepLearningBasedDsaClassification-Validation](https://github.com/sarahelfeel04/DeepLearningBasedDsaClassification-Validation).

## Usage

### Extraction
To extract DSA sequences:
```bash
python extraction/extractDataFromDb.py
```

### Inference
To run classification:
```bash
python inference/classifyAllData.py
```

## Requirements

- Python 3.x
- PyTorch
- NumPy
- nibabel
- pickle
- lmdb
- Other dependencies as listed in the scripts