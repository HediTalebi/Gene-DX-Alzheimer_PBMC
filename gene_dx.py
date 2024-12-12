"""
#import libraries
"""

import os
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import torchmetrics
from sklearn import preprocessing
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score,
                             average_precision_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
!git clone https://github.com/m0hssn/Metrica.git
from Metrica.metrica import Metrica

"""#Data"""

df = pd.read_csv('/content/T_cell_for_auto_train.csv')

df['label'] = df['Unnamed: 0'].str[:2]

df['label'] = df['label'].replace({'AD': 0, 'NC': 1})

df2 = df.copy(deep=True)

cols_to_drop=["Unnamed: 0"]
df2= df2.drop(cols_to_drop, axis=1)

cols_to_drop = ["label"]

filtered_columns = [col for col in df2.columns if not col.startswith('RPS')]

df2 = df2[filtered_columns]

train_x = df2.drop(cols_to_drop, axis=1)

train_y = df2["label"]

dft =pd.read_csv("/content/T_cell_for_auto_test.csv")

filtered_columns = [col for col in dft.columns if not col.startswith('RPL')]

dft['label'] = dft['Unnamed: 0'].str[:2]

dft['label'] = dft['label'].replace({'AD': 0, 'NC': 1})

dft2 = dft.copy(deep=True)

cols_to_drop=["Unnamed: 0"]
dft2= dft2.drop(cols_to_drop, axis=1)

cols_to_drop = ["label"]

test_x = dft2.drop(cols_to_drop, axis=1)

test_y = dft2[cols_to_drop]

"""#model"""

import torch
import torch.nn as nn

class AdversarialAutoencoder(nn.Module):
    def __init__(self, Input_size, label_size):
        super(AdversarialAutoencoder, self).__init__()
        self.input_size = Input_size
        self.label_size = label_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1)  # Added dropout
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),  # Adjusted input size from 256 to 128
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added dropout
            nn.Linear(2048, self.input_size),
            nn.Sigmoid()
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Adjusted the number of units
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # Adjusted input size from 128+2 to 128
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 128),  # Adjusted the number of units
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.label_size),
            nn.LogSoftmax()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)

        random_latent = torch.randn_like(encoded)

        real_output = self.discriminator(encoded)
        fake_output = self.discriminator(random_latent)

        return decoded, classification, real_output, fake_output

"""#train"""

Input_size = df2.shape[1] - 1
label_size = 2

from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, TensorDataset


class CrossValidatedAdversarialAutoencoderTrainer:
    def __init__(self, df2, n_folds=5, batch_size=64, lr=3e-6, num_epochs=30, alpha=0.5, betha=1, gama=1):
        self.df2 = df2
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.best_acc = 0
        self.alpha = alpha
        self.betha = betha
        self.gama = gama

    def train_cross_validated(self):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=7)
        fold_models = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(self.df2)):
            print(f"Training Fold {fold + 1}...")

            train_x, test_x = self.df2.iloc[train_idx], self.df2.iloc[test_idx]

            # ... (rest of your existing code for data preparation)
            label_ctrain = train_x['label'].value_counts()
            print(label_ctrain)

            label_ctest = test_x['label'].value_counts()
            print(label_ctest)

            # Convert data to PyTorch tensors
            train_x_tensor = torch.tensor(train_x.drop('label', axis=1).values, dtype=torch.float32)
            train_y_tensor = torch.tensor(train_x['label'].values, dtype=torch.long)
            test_x_tensor = torch.tensor(test_x.drop('label', axis=1).values, dtype=torch.float32)
            test_y_tensor = torch.tensor(test_x['label'].values, dtype=torch.long)

            # Create PyTorch DataLoader
            train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


            # Initialize AAE model
            model = AdversarialAutoencoder(Input_size, label_size)
            decoder_criterion = nn.L1Loss()
            adversarial_criterion = nn.BCELoss()
            classifier_criterion = nn.NLLLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold=0.003, cooldown=2)

            # Train the model
            best_accuracy = self.train_adversarial_autoencoder(model, train_loader, test_loader, decoder_criterion, classifier_criterion, adversarial_criterion, optimizer, scheduler, self.num_epochs, self.best_acc, self.alpha, self.betha, self.gama)

            fold_models.append(model)

        return fold_models

    def train_adversarial_autoencoder(self, model, train_loader, test_loader, decoder_criterion, classifier_criterion, adversarial_criterion, optimizer, scheduler, num_epochs, best_acc, alpha, betha, gama):
        # ... (rest of your existing code for training)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        best_accuracy = best_acc
        model = model.to(device)
        for epoch in range(num_epochs):
            model.train()
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (training)", leave=False)
            running_loss = 0.0
            running_decoder_loss = 0.0
            running_classifier_loss = 0.0
            running_correct = 0
            running_total = 0

            test_metrica = Metrica(num_classes=2)

            for i, (inputs, labels) in enumerate(train_pbar):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                reconstructed_inputs, predicted_labels, real_output, fake_output = model(inputs)
                decoder_loss = decoder_criterion(reconstructed_inputs, inputs)
                classifier_loss = classifier_criterion(predicted_labels, labels)
                real_labels = torch.ones_like(real_output)
                fake_labels = torch.zeros_like(fake_output)
                adversarial_loss = adversarial_criterion(real_output, real_labels) + adversarial_criterion(fake_output, fake_labels)

                loss = alpha * decoder_loss + gama * classifier_loss + betha * adversarial_loss

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_decoder_loss += decoder_loss.item()
                running_classifier_loss += classifier_loss.item()
                _, predicted = torch.max(predicted_labels.data, 1)
                running_total += labels.size(0)
                running_correct += (predicted == labels).sum().item()
                train_pbar.set_postfix({"loss": running_loss / (i+1),
                                        "decoder_loss": running_decoder_loss / (i+1),
                                        "classifier_loss": running_classifier_loss / (i+1),
                                        "accuracy": 100 * running_correct / running_total})

                optimizer.zero_grad()
                random_latent = torch.randn_like(real_output)
                real_loss = adversarial_criterion(real_output, real_labels)
                fake_loss = adversarial_criterion(fake_output, fake_labels)
                discriminator_loss = real_loss + fake_loss

                optimizer.step()

            epoch_loss = running_loss / len(train_loader)
            epoch_decoder_loss = running_decoder_loss / len(train_loader)
            epoch_classifier_loss = running_classifier_loss / len(train_loader)
            epoch_train_accuracy = 100 * running_correct / running_total
            train_pbar.set_postfix({"loss": epoch_loss,
                                    "decoder_loss": epoch_decoder_loss,
                                    "classifier_loss": epoch_classifier_loss,
                                    "accuracy": epoch_train_accuracy})

            model.eval()
            test_running_loss = 0.0
            test_running_correct = 0
            test_running_total = 0
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} (testing)", leave=False)
            with torch.no_grad():
                for inputs, labels in test_pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    reconstructed_inputs, predicted_labels, _, _ = model(inputs)

                    test_metrica.upgrade(predicted_labels, labels)

                    decoder_loss = decoder_criterion(reconstructed_inputs, inputs)
                    classifier_loss = classifier_criterion(predicted_labels, labels)
                    loss = decoder_loss + classifier_loss
                    # loss = classifier_loss
                    test_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(predicted_labels.data, 1)
                    test_running_total += labels.size(0)
                    test_running_correct += (predicted == labels).sum().item()
                test_accuracy = 100 * test_running_correct / test_running_total
                test_loss = test_running_loss / len(test_loader.dataset)
            test_pbar.set_postfix({"loss": test_loss,
                                  "accuracy": test_accuracy})
            scheduler.step(test_accuracy)
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Autoencoder Loss: {:.4f}, Train Classification Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(epoch+1, num_epochs, epoch_loss, epoch_decoder_loss, epoch_classifier_loss, epoch_train_accuracy, test_loss, test_accuracy))

            test_metrica.print_metrics()
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), "best_model_AAE.pth")

        return best_accuracy

trainer = CrossValidatedAdversarialAutoencoderTrainer(df2, n_folds=5, batch_size=64, lr=3e-6, num_epochs=30, alpha=0.5, betha=1, gama=1)
fold_models = trainer.train_cross_validated()

"""#Matrix"""

# Extract the weights of the encoder and classifier
encoder_weights = model.encoder.state_dict()
classifier_weights = model.classifier.state_dict()

# Convert the weights to NumPy arrays
encoder_weights_np = {key: value.cpu().numpy() for key, value in encoder_weights.items()}
classifier_weights_np = {key: value.cpu().numpy() for key, value in classifier_weights.items()}

# Save the weights as NumPy arrays
np.save('encoder_weights.npy', encoder_weights_np)
np.save('classifier_weights.npy', classifier_weights_np)

# Save encoder weights
for i, (name, param) in enumerate(model.encoder.named_parameters()):
    if 'weight' in name:
        np.save(f'encoder_weight_layer_{i}.npy', param.detach().cpu().numpy())

# Save classifier weights
for i, (name, param) in enumerate(model.classifier.named_parameters()):
    if 'weight' in name:
        np.save(f'classifier_weight_layer_{i}.npy', param.detach().cpu().numpy())

# load the weights
encoder_layer_1 = np.load("encoder_weight_layer_0.npy")
encoder_layer_2 = np.load("encoder_weight_layer_2.npy")
encoder_layer_3 = np.load("encoder_weight_layer_4.npy")
encoder_layer_4 = np.load("encoder_weight_layer_6.npy")
encoder_layer_5 = np.load("encoder_weight_layer_8.npy")
encoder_layer_6 = np.load("encoder_weight_layer_10.npy")
encoder_layer_7 = np.load("encoder_weight_layer_12.npy")


print("encoder_layer1 shape", encoder_layer_1.shape)
print("encoder_layer2 shape", encoder_layer_2.shape)
print("encoder_layer3 shape", encoder_layer_3.shape)
print("encoder_layer4 shape", encoder_layer_4.shape)
print("encoder_layer5 shape", encoder_layer_5.shape)
print("encoder_layer6 shape", encoder_layer_6.shape)
print("encoder_layer7 shape", encoder_layer_7.shape)

encoder_layer1_layer3_multiplication = np.matmul(encoder_layer_3, encoder_layer_1)
print("Shape of encoder_layer1_layer3_multiplication:", encoder_layer1_layer3_multiplication.shape)

encoder_layer5_layer7_multiplication = np.matmul(encoder_layer_7, encoder_layer_5)
print("Shape of encoder_layer5_layer7_multiplication:", encoder_layer5_layer7_multiplication.shape)

encoder_layer5_layer7_multiplication_encoder9 = np.matmul(encoder_layer_9, encoder_layer5_layer7_multiplication)
print("Shape of encoder_layer5_layer7_multiplication:", encoder_layer5_layer7_multiplication.shape)

encoder_layer5_layer7_multiplication_encoder9_e13 = np.matmul(encoder_layer5_layer7_multiplication_encoder9, encoder_layer1_layer3_multiplication)
print("Shape of encoder_layer5_layer7_multiplication_encoder9_e13:", encoder_layer5_layer7_multiplication_encoder9_e13.shape)

encoder_final = np.matmul( encoder_layer5_layer7_multiplication,encoder_layer1_layer3_multiplication)

# load the weights
classifier_layer_1 = np.load("classifier_weight_layer_0.npy")
classifier_layer_2 = np.load("classifier_weight_layer_2.npy")
classifier_layer_3 = np.load("classifier_weight_layer_4.npy")
classifier_layer_4 = np.load("classifier_weight_layer_6.npy")
classifier_layer_5 = np.load("classifier_weight_layer_8.npy")

print("eclassifier_layer_1 shape", classifier_layer_1.shape)
print("classifier_layer_2 shape", classifier_layer_2.shape)
print("classifier_layer_3 shape", classifier_layer_3.shape)
print("classifier_layer_4 shape", classifier_layer_4.shape)
print("classifier_layer_5 shape", classifier_layer_5.shape)

classifier_layer1_layer3_multiplication = np.matmul(classifier_layer_3, classifier_layer_1)
calssifier_final = np.matmul(classifier_layer_5, classifier_layer1_layer3_multiplication)

Final_results = np.matmul(calssifier_final, encoder_final)
Final_results = Final_results.T

df3 = df2.copy(deep=True)

df3 = df3.drop("label", axis=1)

gene_names = df3.columns

# Convert the transposed gene names array to a DataFrame
gene_names_df = pd.DataFrame(gene_names, columns=['Gene Names'])

gene_names_df.to_csv("M_non_mt_rpl_genes.csv")

np.savetxt("Final_results_M_non_mt_rpl.csv", Final_results, delimiter=",")