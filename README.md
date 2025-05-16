# 🩺 **Pneumonia Detection Using CNN**

A **deep learning-based pneumonia detection system** built using TensorFlow/Keras. This project classifies chest X-ray images into two categories: **PNEUMONIA** or **NORMAL**, using a Convolutional Neural Network (CNN).

<br/>

## 📌 **Project Highlights**

- 🧠 Custom CNN model for binary image classification  
- 🩻 Trained on chest X-ray images (NORMAL & PNEUMONIA)  
- 📊 Achieves ~73% test accuracy with a basic CNN architecture  
- ⚙️ Built in Jupyter Notebook using TensorFlow  
- 🗃️ Easy-to-follow and reproducible  

<br/>

## 🧾 **Dataset Source**

The dataset used in this project is the **Chest X-Ray Images (Pneumonia)** dataset, publicly available on **Mendeley Data**:

🔗 [https://data.mendeley.com/datasets/rscbjbr9sj/2](https://data.mendeley.com/datasets/rscbjbr9sj/2)

The dataset includes:
- `train/` (80%): 4185 images  
- `val/` (20%): 1047 images  
- `test/`: 624 images  

Each subset contains:
- `/NORMAL` – healthy X-rays  
- `/PNEUMONIA` – infected X-rays

<br/>

## 🛠️ **Technologies Used**

- Python 3.x  
- Jupyter Notebook  
- **TensorFlow/Keras** for modeling  
- **Pandas**, **NumPy** for data manipulation  
- **Scikit-learn** for dataset splitting  
- **PIL** for image processing  
- **Matplotlib** for plotting  

<br/>

## 🚀 **Installation Instructions**

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
```
### 2. Install Required Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 3. Launch Jupyter Notebook
``` bash
jupyter notebook
```
## 📄 **License**

This project is open-source under the MIT License. see the [MIT License](https://github.com/shreyasrai1/pneumonia-detection-model/blob/main/MIT%20License.txt) file for details.
