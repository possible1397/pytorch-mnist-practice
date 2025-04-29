

🧠 PyTorch MNIST Classifier Practice

This is a beginner-friendly PyTorch project for training a neural network to recognize handwritten digits using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## 📌 Project Features

- ✅ Trains a basic **MLP (Multi-Layer Perceptron)** model
- ✅ Achieves over **94% accuracy** on the MNIST test set
- ✅ Includes a 3-layer MLP visualization script to understand hidden layers and ReLU activation
- ✅ Cleans up unnecessary files using `.gitignore` for a clean GitHub repo

## 📂 Project Structure
 ├── mnist_train.py # Main training script (MLP on MNIST) ├── pytorch 3nn.py # Toy example of 3-layer NN with ReLU and visualization ├── .gitignore # Files/folders excluded from version control
## 🚀 How to Run

Make sure you have Python + PyTorch installed.

Install dependencies (if not already):
```bash
pip install torch torchvision matplotlib

Run the training script:
python mnist_train.py

Optional: Visualize the 3-layer network behavior with:
python "pytorch 3nn.py"

🧠 Learning Focus
This project is designed to practice:

How neural networks learn from data

The structure of feedforward networks (MLPs)

How ReLU activation transforms intermediate layers

Using torch, nn, DataLoader, optimizer, and loss

🛠 To-Do / Future Work
 Convert model to CNN for higher accuracy

 Save and load .pt model files

 Add evaluation report and confusion matrix

 Deploy as a web app using Gradio or Streamlit

🙌 Credits
Created by Terry for PyTorch training practice.