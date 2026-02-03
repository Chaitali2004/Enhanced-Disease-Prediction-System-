# %%
"""
Enhanced Disease Prediction System
Multi-modal deep learning framework combining contrastive learning,
graph neural networks, and hierarchical attention for symptom-based diagnosis.

Author: Chaitali Jain
Major Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (confusion_matrix, classification_report, 
                             ConfusionMatrixDisplay, precision_recall_fscore_support)
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Concatenate, BatchNormalization,
                                      MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
                                      Reshape, Conv1D, MaxPooling1D, Flatten, Add, Lambda)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import regularizers
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import gensim.downloader as api
import warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_PATH = r"C:\Users\CHAITALI JAIN\Desktop\database for eds\DiseaseAndSymptoms.csv"
TRAIN_PATH = r"C:\Users\CHAITALI JAIN\Desktop\database for eds\DiseaseAndSymptoms_train.csv"
TEST_PATH = r"C:\Users\CHAITALI JAIN\Desktop\database for eds\DiseaseAndSymptoms_test.csv"

symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]
embedding_name = "glove-wiki-gigaword-50"
random_state = 42
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

print("="*80)
print("Enhanced Disease Prediction System")
print("="*80)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

df = pd.read_csv(DATA_PATH)
print(f"\nDataset: {df.shape[0]} samples, {df['Disease'].nunique()} diseases")

# Train-test split with stratification
train_df, test_df = train_test_split(df, test_size=0.3, random_state=random_state, 
                                      stratify=df['Disease'])
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

df_train = train_df.copy()
df_test = test_df.copy()
label_col = 'Disease'

# Preprocess symptom columns
for col in symptom_cols:
    if col not in df_train.columns:
        df_train[col] = 'none'
    df_train[col] = df_train[col].fillna('none').astype(str)

# Extract symptom lists from each patient record
symptom_lists = df_train[symptom_cols].apply(
    lambda row: [str(s).strip() for s in row if str(s).strip().lower() not in ['', 'none', 'nan']], 
    axis=1
)

# ============================================================================
# SYMPTOM SEVERITY MAPPING
# ============================================================================
# Map symptoms to severity scores based on frequency distribution

all_symptoms_flat = [s.lower() for lst in symptom_lists for s in lst]
symptom_counts = pd.Series(all_symptoms_flat).value_counts()
quantiles = symptom_counts.quantile([0.2, 0.4, 0.6, 0.8]).values

def freq_to_sev(cnt):
    """Convert symptom frequency to severity score (1-5 scale)"""
    if cnt <= quantiles[0]: return 1
    elif cnt <= quantiles[1]: return 2
    elif cnt <= quantiles[2]: return 3
    elif cnt <= quantiles[3]: return 4
    else: return 5

symptom_to_severity = {sym: freq_to_sev(int(cnt)) for sym, cnt in symptom_counts.items()}

def get_symptom_severity_array(symptom_list):
    """Get severity scores for a list of symptoms"""
    return np.array([symptom_to_severity.get(s.lower(), 1) for s in symptom_list], dtype=float)

print(f"Symptom severity mapping: {len(symptom_to_severity)} symptoms")

# ============================================================================
# WORD EMBEDDINGS
# ============================================================================
# Load pre-trained GloVe embeddings for symptom representation

print("\nLoading word embeddings...")
w2v = api.load(embedding_name)
embed_dim = w2v.vector_size

def embed_symptom(symptom_str):
    """Convert symptom text to embedding vector"""
    words = [w for w in str(symptom_str).lower().split() if w in w2v]
    if not words: 
        return np.zeros(embed_dim, dtype=float)
    return np.mean([w2v[w] for w in words], axis=0)

# Create symptom vocabulary and embedding matrix
unique_symptoms = sorted(symptom_counts.index.tolist())
symptom_index = {sym: i for i, sym in enumerate(unique_symptoms)}
n_sym = len(unique_symptoms)

symptom_embeddings = np.vstack([embed_symptom(sym) for sym in unique_symptoms])
print(f"Symptom embeddings: {symptom_embeddings.shape}")

# ============================================================================
# FEATURE EXTRACTION BRANCH 1: CONTRASTIVE LEARNING
# ============================================================================
# Self-supervised pre-training using NT-Xent contrastive loss

print("\n" + "="*80)
print("Branch 1: Contrastive Learning")
print("="*80)

class ContrastiveSymptomEncoder(nn.Module):
    """
    Encoder for self-supervised contrastive learning.
    Pre-trains symptom representations by learning to separate different diseases
    in embedding space using NT-Xent (InfoNCE) loss.
    """
    def __init__(self, embedding_dim=50, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        self.attention = nn.MultiheadAttention(output_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, symptom_embeddings):
        batch_size = symptom_embeddings.size(0)
        seq_len = symptom_embeddings.size(1)
        x = symptom_embeddings.view(-1, symptom_embeddings.size(-1))
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, seq_len, -1)
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = self.layer_norm(attended + encoded)
        patient_emb = attended.mean(dim=1)
        return encoded, patient_emb

def contrastive_pretrain(encoder, symptom_lists, diseases, epochs=30, batch_size=32):
    """
    Pre-train encoder using NT-Xent contrastive loss.
    Positive pairs: symptoms from same disease
    Negative pairs: symptoms from different diseases
    """
    print("Pre-training contrastive encoder...")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    temperature = 0.07
    
    unique_diseases = diseases.unique()
    disease_to_idx = {d: i for i, d in enumerate(unique_diseases)}
    disease_labels = np.array([disease_to_idx[d] for d in diseases])
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        indices = np.random.permutation(len(symptom_lists))
        
        for i in range(0, len(symptom_lists), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_symptoms = [symptom_lists.iloc[idx] for idx in batch_idx]
            batch_diseases = disease_labels[batch_idx]
            
            max_len = max(len(s) for s in batch_symptoms)
            padded_symptoms = []
            
            for symptoms in batch_symptoms:
                symp_vecs = [embed_symptom(s) for s in symptoms]
                while len(symp_vecs) < max_len:
                    symp_vecs.append(np.zeros(embed_dim))
                padded_symptoms.append(np.stack(symp_vecs))
            
            symptom_tensor = torch.tensor(np.stack(padded_symptoms), dtype=torch.float32)
            _, patient_embeddings = encoder(symptom_tensor)
            
            # Compute NT-Xent loss
            loss = 0
            valid_pairs = 0
            
            for j in range(len(patient_embeddings)):
                pos_mask = (batch_diseases == batch_diseases[j])
                pos_mask[j] = False
                
                if pos_mask.sum() > 0:
                    anchor = patient_embeddings[j].unsqueeze(0)
                    similarities = F.cosine_similarity(anchor, patient_embeddings, dim=1) / temperature
                    labels = torch.tensor(pos_mask.astype(float), dtype=torch.float32)
                    exp_sim = torch.exp(similarities)
                    pos_sum = (exp_sim * labels).sum()
                    total_sum = exp_sim.sum()
                    
                    if pos_sum > 0:
                        loss += -torch.log(pos_sum / (total_sum + 1e-8))
                        valid_pairs += 1
            
            if valid_pairs > 0:
                loss = loss / valid_pairs
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print("Pre-training complete")
    return encoder

# Initialize and pre-train encoder
contrastive_encoder = ContrastiveSymptomEncoder(embed_dim, 128, 64)
contrastive_encoder = contrastive_pretrain(contrastive_encoder, symptom_lists, 
                                           df_train[label_col], epochs=30)

# Extract contrastive features for training set
print("\nExtracting contrastive features...")
contrastive_encoder.eval()
X_contrastive = []

with torch.no_grad():
    for symptoms in symptom_lists:
        if len(symptoms) == 0:
            X_contrastive.append(np.zeros(64))
            continue
        symp_vecs = torch.tensor(np.stack([embed_symptom(s) for s in symptoms]), 
                                dtype=torch.float32).unsqueeze(0)
        _, patient_emb = contrastive_encoder(symp_vecs)
        X_contrastive.append(patient_emb.squeeze(0).numpy())

X_contrastive = np.array(X_contrastive)
print(f"Contrastive features: {X_contrastive.shape}")

# ============================================================================
# FEATURE EXTRACTION BRANCH 2: GRAPH NEURAL NETWORK
# ============================================================================
# Graph attention network with medical knowledge integration

print("\n" + "="*80)
print("Branch 2: Graph Neural Network")
print("="*80)

# Body system taxonomy for medical knowledge integration
body_systems = {
    'respiratory': ['cough', 'breathlessness', 'chest_pain', 'wheezing', 'phlegm', 
                    'throat_irritation', 'sinus_pressure', 'runny_nose', 'congestion'],
    'digestive': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'constipation',
                  'stomach_pain', 'indigestion', 'loss_of_appetite', 'bloating', 
                  'acidity', 'ulcers_on_tongue'],
    'nervous': ['headache', 'dizziness', 'confusion', 'seizures', 'anxiety',
                'depression', 'mood_swings', 'lack_of_concentration', 'unsteadiness',
                'altered_sensorium', 'visual_disturbances'],
    'circulatory': ['palpitations', 'chest_pain', 'fatigue', 'irregular_heartbeat',
                    'high_blood_pressure', 'swelling', 'cold_hands_and_feets',
                    'prominent_veins_on_calf'],
    'musculoskeletal': ['muscle_weakness', 'joint_pain', 'neck_pain', 'knee_pain',
                        'back_pain', 'muscle_pain', 'stiff_neck', 'pain_in_anal_region',
                        'cramps', 'movement_stiffness'],
    'dermatological': ['skin_rash', 'itching', 'yellowish_skin', 'red_spots_over_body',
                       'dischromic_patches', 'bruising', 'yellow_crust_ooze', 
                       'nodal_skin_eruptions', 'pus_filled_pimples', 'blackheads'],
    'urinary': ['burning_micturition', 'bladder_discomfort', 'continuous_feel_of_urine',
                'dark_urine', 'yellow_urine', 'polyuria'],
    'immune': ['fever', 'chills', 'shivering', 'sweating', 'malaise', 'lethargy',
               'mild_fever', 'high_fever'],
    'endocrine': ['weight_loss', 'weight_gain', 'excessive_hunger', 'increased_appetite',
                  'obesity', 'enlarged_thyroid', 'brittle_nails', 'abnormal_menstruation']
}

def build_symptom_graph(symptom_embeddings, symptom_counts):
    """
    Construct weighted symptom graph using three factors:
    1. Semantic similarity (from embeddings)
    2. Body system connectivity (medical knowledge)
    3. Co-occurrence patterns (data-driven)
    """
    print("Building symptom graph...")
    
    # Factor 1: Semantic similarity
    symptom_sim = cosine_similarity(symptom_embeddings)
    
    # Factor 2: Co-occurrence matrix
    cooc = np.zeros((n_sym, n_sym))
    for lst in symptom_lists:
        idxs = [symptom_index[s.lower()] for s in lst if s.lower() in symptom_index]
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                cooc[idxs[i], idxs[j]] += 1
                cooc[idxs[j], idxs[i]] += 1
    cooc_max = cooc.max() if cooc.max() > 0 else 1
    cooc_norm = cooc / cooc_max
    
    # Factor 3: Body system connectivity
    body_system_matrix = np.zeros((n_sym, n_sym))
    for system, symptoms in body_systems.items():
        system_indices = []
        for sym in symptoms:
            for unique_sym in unique_symptoms:
                if sym in unique_sym or unique_sym in sym:
                    system_indices.append(symptom_index[unique_sym])
                    break
        for i in system_indices:
            for j in system_indices:
                if i != j:
                    body_system_matrix[i, j] = 0.8
    
    # Combine factors with weighted sum
    edges = []
    edge_weights = []
    
    for i in range(n_sym):
        for j in range(i+1, n_sym):
            weight = 0
            if symptom_sim[i, j] > 0.3:
                weight += symptom_sim[i, j] * 0.3
            weight += body_system_matrix[i, j] * 0.4
            weight += cooc_norm[i, j] * 0.3
            
            if weight > 0.15:
                edges.append([i, j])
                edges.append([j, i])
                edge_weights.extend([weight, weight])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    print(f"Graph: {n_sym} nodes, {len(edges)} edges")
    return edge_index, edge_weight

class SymptomGNN(nn.Module):
    """
    Graph Attention Network for symptom propagation.
    Uses multi-layer GAT to learn symptom relationships through graph structure.
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=3, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, dropout=0.3))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.3))
        self.output_dim = hidden_dim * heads
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.input_proj(x))
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=0.3, training=self.training)
        return x

# Build graph and compute node embeddings
edge_index, edge_weight = build_symptom_graph(symptom_embeddings, symptom_counts)

gnn_model = SymptomGNN(input_dim=embed_dim, hidden_dim=32, num_layers=3, heads=4)
gnn_model.eval()

print("Computing graph embeddings...")
with torch.no_grad():
    node_features = torch.tensor(symptom_embeddings, dtype=torch.float32)
    graph_embeddings = gnn_model(node_features, edge_index, edge_weight).numpy()

print(f"Graph embeddings: {graph_embeddings.shape}")

def patient_graph_features(symptom_list):
    """Aggregate graph embeddings for patient with severity weighting"""
    idxs = [symptom_index[s.lower()] for s in symptom_list if s.lower() in symptom_index]
    if not idxs:
        return np.zeros(gnn_model.output_dim)
    severities = np.array([symptom_to_severity.get(s.lower(), 1) for s in symptom_list 
                          if s.lower() in symptom_index])
    severities = severities / severities.sum()
    return (graph_embeddings[idxs] * severities[:, None]).sum(axis=0)

X_gnn = np.vstack([patient_graph_features(lst) for lst in symptom_lists])
print(f"GNN features: {X_gnn.shape}")

# ============================================================================
# FEATURE EXTRACTION BRANCH 3: HIERARCHICAL ATTENTION
# ============================================================================
# Multi-level attention mechanism (word-level + phrase-level)

print("\n" + "="*80)
print("Branch 3: Multi-Level Attention")
print("="*80)

def multi_level_attention(symptom_list):
    """
    Compute hierarchical attention at two levels:
    - Word level: Individual symptom importance
    - Phrase level: Symptom pair/combination importance
    Both weighted by severity scores.
    """
    if len(symptom_list) == 0:
        return np.zeros(embed_dim * 2)
    
    # Level 1: Word-level attention
    vecs = np.vstack([embed_symptom(s) for s in symptom_list])
    context = vecs.mean(axis=0)
    scores = vecs.dot(context)
    exp_scores = np.exp(scores - np.max(scores))
    attn = exp_scores / (np.sum(exp_scores) + 1e-12)
    sev = get_symptom_severity_array(symptom_list)
    sev_norm = 0.5 + (sev - 1) / 4.0
    combined_weights = attn * sev_norm
    combined_weights /= (combined_weights.sum() + 1e-12)
    word_level = (vecs * combined_weights[:, None]).sum(axis=0)
    
    # Level 2: Phrase-level attention
    phrases = [f"{symptom_list[i]} {symptom_list[i+1]}" 
               for i in range(len(symptom_list) - 1)]
    
    if len(phrases) == 0:
        phrase_level = np.zeros(embed_dim)
    else:
        phrase_vecs = np.vstack([embed_symptom(p) for p in phrases])
        phrase_context = phrase_vecs.mean(axis=0)
        phrase_scores = phrase_vecs.dot(phrase_context)
        phrase_exp = np.exp(phrase_scores - np.max(phrase_scores))
        phrase_attn = phrase_exp / (np.sum(phrase_exp) + 1e-12)
        phrase_level = (phrase_vecs * phrase_attn[:, None]).sum(axis=0)
    
    return np.hstack([word_level, phrase_level])

print("Computing attention features...")
X_attention = np.vstack([multi_level_attention(lst) for lst in symptom_lists])
print(f"Attention features: {X_attention.shape}")

# ============================================================================
# FEATURE EXTRACTION BRANCH 4: STRUCTURED FEATURES
# ============================================================================
# One-hot encoding of top-50 most frequent symptoms

mlb = MultiLabelBinarizer()
X_structured_full = mlb.fit_transform(symptom_lists)
top_symptoms = symptom_counts.head(50).index.tolist()
top_idx = [mlb.classes_.tolist().index(s) for s in top_symptoms]
X_structured = X_structured_full[:, top_idx]
print(f"\nStructured features: {X_structured.shape}")

# ============================================================================
# FEATURE SCALING AND LABEL ENCODING
# ============================================================================

scaler_contrastive = StandardScaler()
X_contrastive_scaled = scaler_contrastive.fit_transform(X_contrastive)

scaler_gnn = StandardScaler()
X_gnn_scaled = scaler_gnn.fit_transform(X_gnn)

scaler_attention = StandardScaler()
X_attention_scaled = scaler_attention.fit_transform(X_attention)

print(f"\nFeature summary:")
print(f"  Contrastive: {X_contrastive_scaled.shape[1]}D")
print(f"  GNN: {X_gnn_scaled.shape[1]}D")
print(f"  Attention: {X_attention_scaled.shape[1]}D")
print(f"  Structured: {X_structured.shape[1]}D")

le = LabelEncoder()
y = le.fit_transform(df_train[label_col].astype(str).values)
y_train_cat = to_categorical(y, len(le.classes_))
num_classes = len(le.classes_)
print(f"Disease classes: {num_classes}")

# ============================================================================
# PROCESS TEST SET
# ============================================================================

print("\nProcessing test set...")
for col in symptom_cols:
    if col not in df_test.columns:
        df_test[col] = 'none'
    df_test[col] = df_test[col].fillna('none').astype(str)

symptom_lists_test = df_test[symptom_cols].apply(
    lambda row: [str(s).strip() for s in row if str(s).strip().lower() not in ['', 'none', 'nan']], 
    axis=1
)

# Extract all test features
X_contrastive_test = []
with torch.no_grad():
    for symptoms in symptom_lists_test:
        if len(symptoms) == 0:
            X_contrastive_test.append(np.zeros(64))
            continue
        symp_vecs = torch.tensor(np.stack([embed_symptom(s) for s in symptoms]), 
                                dtype=torch.float32).unsqueeze(0)
        _, patient_emb = contrastive_encoder(symp_vecs)
        X_contrastive_test.append(patient_emb.squeeze(0).numpy())
X_contrastive_test = np.array(X_contrastive_test)

X_gnn_test = np.vstack([patient_graph_features(lst) for lst in symptom_lists_test])
X_attention_test = np.vstack([multi_level_attention(lst) for lst in symptom_lists_test])
X_structured_test_full = mlb.transform(symptom_lists_test)
X_structured_test = X_structured_test_full[:, top_idx]

X_contrastive_scaled_test = scaler_contrastive.transform(X_contrastive_test)
X_gnn_scaled_test = scaler_gnn.transform(X_gnn_test)
X_attention_scaled_test = scaler_attention.transform(X_attention_test)

y_test = le.transform(df_test[label_col].astype(str).values)
y_test_cat = to_categorical(y_test, num_classes)

# ============================================================================
# BUILD MULTI-MODAL FUSION MODEL
# ============================================================================

print("\n" + "="*80)
print("Building Multi-Modal Fusion Model")
print("="*80)

contrastive_dim = X_contrastive_scaled.shape[1]
gnn_dim = X_gnn_scaled.shape[1]
att_dim = X_attention_scaled.shape[1]
struct_dim = X_structured.shape[1]

l2_reg = regularizers.l2(0.015)

# Branch 1: Contrastive features
contrastive_input = Input(shape=(contrastive_dim,), name='contrastive_input')
x1 = Dense(128, activation='relu', kernel_regularizer=l2_reg)(contrastive_input)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.45)(x1)
x1 = Dense(64, activation='relu', kernel_regularizer=l2_reg)(x1)
x1 = Dropout(0.35)(x1)

# Branch 2: GNN features
gnn_input = Input(shape=(gnn_dim,), name='gnn_input')
x2 = Dense(128, activation='relu', kernel_regularizer=l2_reg)(gnn_input)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.45)(x2)
x2 = Dense(64, activation='relu', kernel_regularizer=l2_reg)(x2)
x2 = Dropout(0.35)(x2)

# Branch 3: Attention with multi-head attention mechanism
att_input = Input(shape=(att_dim,), name='attention_input')
att_reshaped = Reshape((1, att_dim))(att_input)
att_mha = MultiHeadAttention(num_heads=4, key_dim=att_dim//4)(att_reshaped, att_reshaped)
att_mha = LayerNormalization()(att_mha)
att_pooled = GlobalAveragePooling1D()(att_mha)
x3 = Dense(64, activation='relu', kernel_regularizer=l2_reg)(att_pooled)
x3 = BatchNormalization()(x3)
x3 = Dropout(0.45)(x3)

# Branch 4: Structured features with 1D CNN
struct_input = Input(shape=(struct_dim,), name='structured_input')
struct_reshaped = Reshape((struct_dim, 1))(struct_input)
struct_conv = Conv1D(32, kernel_size=3, activation='relu', padding='same', 
                     kernel_regularizer=l2_reg)(struct_reshaped)
struct_conv = Dropout(0.35)(struct_conv)
struct_pool = MaxPooling1D(pool_size=2)(struct_conv)
struct_conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', 
                      kernel_regularizer=l2_reg)(struct_pool)
struct_flat = Flatten()(struct_conv2)
x4 = Dense(64, activation='relu', kernel_regularizer=l2_reg)(struct_flat)
x4 = Dropout(0.35)(x4)

# Multi-modal fusion with residual connection
merged = Concatenate()([x1, x2, x3, x4])
fusion = Dense(256, activation='relu', kernel_regularizer=l2_reg)(merged)
fusion = BatchNormalization()(fusion)
fusion = Dropout(0.55)(fusion)
fusion = Dense(128, activation='relu', kernel_regularizer=l2_reg)(fusion)
fusion = BatchNormalization()(fusion)
fusion = Dropout(0.45)(fusion)

# Residual skip connection
fusion_skip = Dense(128, activation='relu', kernel_regularizer=l2_reg)(merged)
fusion = Add()([fusion, fusion_skip])
fusion = LayerNormalization()(fusion)

fusion = Dense(64, activation='relu', kernel_regularizer=l2_reg)(fusion)
fusion = Dropout(0.3)(fusion)
output = Dense(num_classes, activation='softmax')(fusion)

# Compile model
model = Model(inputs=[contrastive_input, gnn_input, att_input, struct_input],
              outputs=output, name='Disease_Prediction_System')

model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss=CategoricalCrossentropy(label_smoothing=0.2),
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()
print(f"\nTotal parameters: {model.count_params():,}")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
print("Training Model")
print("="*80)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)

history = model.fit(
    [X_contrastive_scaled, X_gnn_scaled, X_attention_scaled, X_structured],
    y_train_cat,
    validation_split=0.25,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\nTraining complete")

# ============================================================================
# EVALUATE MODEL
# ============================================================================

print("\n" + "="*80)
print("Model Evaluation")
print("="*80)

test_loss, test_acc = model.evaluate(
    [X_contrastive_scaled_test, X_gnn_scaled_test, X_attention_scaled_test, X_structured_test],
    y_test_cat, verbose=0
)

print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
y_pred_probs = model.predict([X_contrastive_scaled_test, X_gnn_scaled_test, 
                              X_attention_scaled_test, X_structured_test], verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Top-k accuracy metrics
def top_k_accuracy(y_true, y_pred_probs, k):
    """Calculate top-k accuracy"""
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true[:, None], axis=1)
    return correct.mean()

top3_acc = top_k_accuracy(y_test, y_pred_probs, 3)
top5_acc = top_k_accuracy(y_test, y_pred_probs, 5)

print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")
print(f"Top-5 Accuracy: {top5_acc*100:.2f}%")

# Classification report
print("\n" + "-"*80)
print("Classification Report")
print("-"*80)
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# ============================================================================
# PREDICTION INTERFACE
# ============================================================================

print("\n" + "="*80)
print("Prediction Interface")
print("="*80)

def predict_disease(symptom_list, top_k=3):
    """
    Predict disease from symptom list with top-k predictions.
    
    Parameters:
        symptom_list: List of symptom strings
        top_k: Number of top predictions to return
    
    Returns:
        List of (disease, probability) tuples
    """
    # Extract features
    if len(symptom_list) == 0:
        cont_vec = np.zeros(64)
    else:
        symp_vecs = torch.tensor(np.stack([embed_symptom(s) for s in symptom_list]), 
                                dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, patient_emb = contrastive_encoder(symp_vecs)
        cont_vec = patient_emb.squeeze(0).numpy()
    cont_vec_scaled = scaler_contrastive.transform(cont_vec.reshape(1, -1))
    
    gnn_vec = patient_graph_features(symptom_list)
    gnn_vec_scaled = scaler_gnn.transform(gnn_vec.reshape(1, -1))
    
    att_vec = multi_level_attention(symptom_list)
    att_vec_scaled = scaler_attention.transform(att_vec.reshape(1, -1))
    
    struct_vec_full = mlb.transform([symptom_list])
    struct_vec = struct_vec_full[:, top_idx]
    
    # Predict
    pred_prob = model.predict([cont_vec_scaled, gnn_vec_scaled, att_vec_scaled, struct_vec], verbose=0)
    
    # Get top-k
    top_k_idx = np.argsort(pred_prob[0])[::-1][:top_k]
    
    print(f"\nSymptoms: {symptom_list}")
    print("-"*60)
    print(f"{'Rank':<6} {'Disease':<30} {'Confidence':<12}")
    print("-"*60)
    for i, idx in enumerate(top_k_idx, 1):
        disease = le.classes_[idx]
        confidence = pred_prob[0][idx] * 100
        print(f"{i:<6} {disease:<30} {confidence:>6.2f}%")
    print("-"*60)
    
    return [(le.classes_[idx], pred_prob[0][idx]) for idx in top_k_idx]

# Example predictions
print("\nExample Predictions:")
predict_disease(["fever", "cough", "fatigue"])
predict_disease(["joint_pain", "skin_rash", "nausea"])
predict_disease(["chest_pain", "breathlessness", "dizziness"])

# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================

print("\n" + "="*80)
print("Performance Summary")
print("="*80)

precision_weighted = precision_recall_fscore_support(y_test, y_pred, average='weighted')[0]
recall_weighted = precision_recall_fscore_support(y_test, y_pred, average='weighted')[1]
f1_weighted = precision_recall_fscore_support(y_test, y_pred, average='weighted')[2]

summary = f"""
{'='*80}
                    Disease Prediction System
                        Performance Summary
{'='*80}

Dataset Statistics:
  Disease Classes:     {num_classes}
  Training Samples:    {len(train_df)}
  Test Samples:        {len(test_df)}
  Feature Dimension:   {X_contrastive_scaled.shape[1] + X_gnn_scaled.shape[1] + X_attention_scaled.shape[1] + X_structured.shape[1]}D

Model Architecture:
  Total Parameters:    {model.count_params():,}
  Branches:            4 (Contrastive, GNN, Attention, Structured)
  Pre-training:        30 epochs (NT-Xent loss)
  Graph Network:       3-layer GAT with 4 attention heads
  Fusion:              Multi-modal with residual connections

Performance Metrics:
  Test Accuracy:       {test_acc*100:.2f}%
  Top-3 Accuracy:      {top3_acc*100:.2f}%
  Top-5 Accuracy:      {top5_acc*100:.2f}%
  Weighted Precision:  {precision_weighted*100:.2f}%
  Weighted Recall:     {recall_weighted*100:.2f}%
  Weighted F1-Score:   {f1_weighted*100:.2f}%

Key Features:
  - Self-supervised contrastive pre-training
  - Graph attention networks with body system knowledge
  - Multi-level hierarchical attention (word + phrase)
  - Multi-modal feature fusion

{'='*80}
"""

print(summary)

with open('performance_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("\nSummary saved to 'performance_summary.txt'")


# %%
# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("Generating Visualizations")
print("="*80)

# Plot 1: Training Accuracy
print("Creating training accuracy plot...")
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
plt.title('Training Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('1_training_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 1_training_accuracy.png")

# Plot 2: Training Loss
print("Creating training loss plot...")
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
plt.title('Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2_training_loss.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 2_training_loss.png")

# Plot 3: Top-K Accuracy
print("Creating top-k accuracy plot...")
k_values = [1, 3, 5, 10]
accuracies = [test_acc * 100] + [top_k_accuracy(y_test, y_pred_probs, k) * 100 
                                  for k in k_values[1:]]
plt.figure(figsize=(10, 6))
plt.bar(k_values, accuracies, width=1.5, edgecolor='black', linewidth=1.5)
plt.title('Top-K Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('K')
plt.ylabel('Accuracy (%)')
plt.ylim([0, 105])
plt.grid(True, alpha=0.3, axis='y')
for i, (k, acc) in enumerate(zip(k_values, accuracies)):
    plt.text(k, acc + 2, f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('3_topk_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 3_topk_accuracy.png")

# Plot 4: Confusion Matrix
print("Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
disease_counts = pd.Series(y_test).value_counts().head(10)
top_diseases_idx = disease_counts.index.tolist()
cm_subset = cm[np.ix_(top_diseases_idx, top_diseases_idx)]
top_disease_names = [le.classes_[i] for i in top_diseases_idx]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_subset, annot=True, fmt='d', cmap='YlOrRd', 
           xticklabels=top_disease_names, yticklabels=top_disease_names, 
           cbar_kws={'label': 'Count'}, linewidths=0.5)
plt.title('Confusion Matrix (Top 10 Diseases)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('4_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 4_confusion_matrix.png")

# Plot 4: Confusion Matrix (All Diseases)
print("Creating confusion matrix for all diseases...")

# 1. Compute the full confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 2. Get all disease names from the LabelEncoder
all_disease_names = le.classes_

# 3. Adjust figure size dynamically based on the number of classes
num_classes = len(all_disease_names)
fig_size = max(12, num_classes * 0.4) # Scales size so labels don't overlap

plt.figure(figsize=(fig_size, fig_size * 0.8))

# 4. Create the heatmap
sns.heatmap(
    cm, 
    annot=num_classes < 30, # Only show numbers if the matrix isn't too crowded
    fmt='d', 
    cmap='YlOrRd', 
    xticklabels=all_disease_names, 
    yticklabels=all_disease_names, 
    cbar_kws={'label': 'Count'}, 
    linewidths=0.1
)

plt.title(f'Full Confusion Matrix ({num_classes} Diseases)', fontsize=16, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)

# Rotate labels for better readability
plt.xticks(rotation=90, ha='center', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig('4_confusion_matrix_full.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: 4_confusion_matrix_full.png - Processed {num_classes} classes.")

# Plot 5: Per-Class Performance
print("Creating per-class performance plot...")
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
top_15_idx = np.argsort(support)[-15:][::-1]
diseases_subset = [le.classes_[i][:15] for i in top_15_idx]
x_pos = np.arange(len(diseases_subset))
width = 0.25

plt.figure(figsize=(12, 8))
plt.bar(x_pos - width, precision[top_15_idx], width, label='Precision', alpha=0.8)
plt.bar(x_pos, recall[top_15_idx], width, label='Recall', alpha=0.8)
plt.bar(x_pos + width, f1[top_15_idx], width, label='F1-Score', alpha=0.8)
plt.xlabel('Disease')
plt.ylabel('Score')
plt.title('Per-Class Performance (Top 15 Diseases)', fontsize=14, fontweight='bold')
plt.xticks(x_pos, diseases_subset, rotation=45, ha='right', fontsize=9)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim([0, 1.1])
plt.tight_layout()
plt.savefig('5_perclass_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 5_perclass_performance.png")

# Plot 6: t-SNE Visualization
print("Computing t-SNE projection...")
X_vis = np.hstack([X_contrastive_scaled, X_gnn_scaled, X_attention_scaled])
sample_size = min(1000, len(X_vis))
indices = np.random.choice(len(X_vis), sample_size, replace=False)
X_vis_sample = X_vis[indices]
labels_vis_sample = df_train[label_col].values[indices]

tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
X_tsne = tsne.fit_transform(X_vis_sample)

unique_labels = np.unique(labels_vis_sample)
n_labels = len(unique_labels)
colors = plt.cm.tab20(np.linspace(0, 1, n_labels))

plt.figure(figsize=(14, 10))
for i, label in enumerate(unique_labels):
    mask = labels_vis_sample == label
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[i]], 
               label=label, s=30, alpha=0.6, edgecolors='black', linewidth=0.3)

plt.title('t-SNE Visualization of Patient Embeddings', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('6_tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 6_tsne_visualization.png")

# Plot 7: Attention Heatmap
print("\nGenerating attention heatmap...")
max_symptoms = max(symptom_lists.apply(len))
att_matrix = np.zeros((min(100, len(symptom_lists)), max_symptoms))

for i, lst in enumerate(symptom_lists[:100]):
    if len(lst) == 0:
        continue
    vecs = np.vstack([embed_symptom(s) for s in lst])
    context = vecs.mean(axis=0)
    scores = vecs.dot(context)
    exp_scores = np.exp(scores - np.max(scores))
    attn = exp_scores / (exp_scores.sum() + 1e-12)
    sev = get_symptom_severity_array(lst)
    sev_norm = 0.5 + (sev - 1) / 4
    combined = attn * sev_norm
    combined /= (combined.sum() + 1e-12)
    att_matrix[i, :len(lst)] = combined

plt.figure(figsize=(14, 8))
sns.heatmap(att_matrix, cmap="YlGnBu", cbar_kws={'label': 'Attention Weight'})
plt.xlabel("Symptom Position")
plt.ylabel("Patient Sample")
plt.title("Attention Heatmap with Severity Weighting", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('7_attention_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 7_attention_heatmap.png")

# Plots 8+: Top predictive symptoms per disease (separate file for each)
print("\nAnalyzing top predictive symptoms...")
selected_diseases = ['Tuberculosis', 'Pneumonia', 'Hepatitis B', 'GERD', 'Hypertension']
top_k_symptoms = 5
top_symptoms_selected = {}

for disease in selected_diseases:
    if disease not in le.classes_:
        continue
    class_idx = np.where(le.classes_ == disease)[0][0]
    patient_idxs = np.where(y == class_idx)[0]
    symptom_scores = np.zeros(len(unique_symptoms))
    
    for idx in patient_idxs:
        patient_symptoms = symptom_lists.iloc[idx]
        if len(patient_symptoms) == 0:
            continue
        vecs = np.vstack([embed_symptom(s) for s in patient_symptoms])
        context = vecs.mean(axis=0)
        scores = vecs.dot(context)
        exp_scores = np.exp(scores - np.max(scores))
        attn = exp_scores / (exp_scores.sum() + 1e-12)
        sev = get_symptom_severity_array(patient_symptoms)
        sev_norm = 0.5 + (sev - 1) / 4.0
        combined = attn * sev_norm
        combined = combined / (combined.sum() + 1e-12)
        for s, c in zip(patient_symptoms, combined):
            if s.lower() in symptom_index:
                symptom_scores[symptom_index[s.lower()]] += c
    
    top_idx_symptoms = np.argsort(symptom_scores)[::-1][:top_k_symptoms]
    top_symptoms_selected[disease] = [(unique_symptoms[i], symptom_scores[i]) 
                                       for i in top_idx_symptoms]

# Create separate plot for each disease
for disease_num, (disease, top_symptoms) in enumerate(top_symptoms_selected.items(), 1):
    plt.figure(figsize=(10, 6))
    symptoms, scores = zip(*top_symptoms)
    
    plt.barh(range(len(symptoms)), scores[::-1], edgecolor='black', linewidth=1.5)
    plt.yticks(range(len(symptoms)), symptoms[::-1], fontsize=11)
    plt.xlabel("Cumulative Attention Score", fontsize=11)
    plt.title(f"Top-{top_k_symptoms} Predictive Symptoms: {disease}", 
             fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    disease_filename = disease.replace(" ", "_").replace("/", "_")
    plt.savefig(f'8_{disease_num}_{disease_filename}_symptoms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 8_{disease_num}_{disease_filename}_symptoms.png")

print("\n" + "="*80)
print("All Visualizations Generated Successfully")
print("="*80)
print("\nGenerated Files:")
print("  1. 1_training_accuracy.png")
print("  2. 2_training_loss.png")
print("  3. 3_topk_accuracy.png")
print("  4. 4_confusion_matrix.png")
print("  5. 5_perclass_performance.png")
print("  6. 6_tsne_visualization.png")
print("  7. 7_attention_heatmap.png")
print("  8. 8_1 to 8_5 (top symptoms for each disease)")
print("\n" + "="*80)
print("System Ready")
print("="*80)

# %%
# Ablation Studies
# Tests the contribution of each feature extraction branch
# Updated for Main Model: Contrastive + GNN + Attention + Structured

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Concatenate, BatchNormalization,
                                      MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
                                      Reshape, Conv1D, MaxPooling1D, Flatten, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import regularizers

print("="*80)
print("Ablation Studies - Testing Component Contributions")
print("="*80)

# Assumes you've already run the main code and have these variables:
# X_contrastive_scaled, X_gnn_scaled, X_attention_scaled, X_structured
# X_contrastive_scaled_test, X_gnn_scaled_test, X_attention_scaled_test, X_structured_test
# y_train_cat, y_test_cat, num_classes

l2_reg = regularizers.l2(0.015)

def build_ablation_model(input_dims, input_names, num_classes):
    """Build ablation model with appropriate architecture for each feature type"""
    inputs = []
    branches = []
    
    # Build branches based on feature type
    for i, (dim, name) in enumerate(zip(input_dims, input_names)):
        inp = Input(shape=(dim,), name=f'{name}_input')
        inputs.append(inp)
        
        if 'attention' in name:
            # Attention branch with Multi-Head Attention
            att_reshaped = Reshape((1, dim))(inp)
            att_mha = MultiHeadAttention(num_heads=4, key_dim=dim//4)(att_reshaped, att_reshaped)
            att_mha = LayerNormalization()(att_mha)
            att_pooled = GlobalAveragePooling1D()(att_mha)
            x = Dense(64, activation='relu', kernel_regularizer=l2_reg)(att_pooled)
            x = BatchNormalization()(x)
            x = Dropout(0.45)(x)
            
        elif 'structured' in name:
            # Structured branch with CNN
            struct_reshaped = Reshape((dim, 1))(inp)
            struct_conv = Conv1D(32, kernel_size=3, activation='relu', padding='same', 
                                kernel_regularizer=l2_reg)(struct_reshaped)
            struct_conv = Dropout(0.35)(struct_conv)
            struct_pool = MaxPooling1D(pool_size=2)(struct_conv)
            struct_conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', 
                                 kernel_regularizer=l2_reg)(struct_pool)
            struct_flat = Flatten()(struct_conv2)
            x = Dense(64, activation='relu', kernel_regularizer=l2_reg)(struct_flat)
            x = Dropout(0.35)(x)
            
        else:
            # Standard dense branch (contrastive, GNN)
            x = Dense(128, activation='relu', kernel_regularizer=l2_reg)(inp)
            x = BatchNormalization()(x)
            x = Dropout(0.45)(x)
            x = Dense(64, activation='relu', kernel_regularizer=l2_reg)(x)
            x = Dropout(0.35)(x)
        
        branches.append(x)
    
    if len(branches) > 1:
        merged = Concatenate()(branches)
    else:
        merged = branches[0]
    
    # Fusion layers
    x = Dense(256, activation='relu', kernel_regularizer=l2_reg)(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.55)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    
    # Residual connection
    fusion_skip = Dense(128, activation='relu', kernel_regularizer=l2_reg)(merged)
    x = Add()([x, fusion_skip])
    x = LayerNormalization()(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2_reg)(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss=CategoricalCrossentropy(label_smoothing=0.2),
        metrics=['accuracy']
    )
    return model

# ==================== Test 1: Full Model ====================
print("\n" + "-"*80)
print("Test 1: Full Model (All Features)")
print("-"*80)

full_model = build_ablation_model(
    [X_contrastive_scaled.shape[1], X_gnn_scaled.shape[1], 
     X_attention_scaled.shape[1], X_structured.shape[1]],
    ['contrastive', 'gnn', 'attention', 'structured'],
    num_classes
)

full_model.fit(
    [X_contrastive_scaled, X_gnn_scaled, X_attention_scaled, X_structured],
    y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

_, full_acc = full_model.evaluate(
    [X_contrastive_scaled_test, X_gnn_scaled_test, X_attention_scaled_test, X_structured_test],
    y_test_cat,
    verbose=0
)

print(f"Full Model Accuracy: {full_acc*100:.2f}%")

# ==================== Test 2: Without Contrastive ====================
print("\n" + "-"*80)
print("Test 2: Without Contrastive Features")
print("-"*80)

model_no_contrastive = build_ablation_model(
    [X_gnn_scaled.shape[1], X_attention_scaled.shape[1], X_structured.shape[1]],
    ['gnn', 'attention', 'structured'],
    num_classes
)

model_no_contrastive.fit(
    [X_gnn_scaled, X_attention_scaled, X_structured],
    y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

_, acc_no_contrastive = model_no_contrastive.evaluate(
    [X_gnn_scaled_test, X_attention_scaled_test, X_structured_test],
    y_test_cat,
    verbose=0
)

print(f"Without Contrastive: {acc_no_contrastive*100:.2f}%")
print(f"Drop: {(full_acc - acc_no_contrastive)*100:.2f}%")

# ==================== Test 3: Without GNN ====================
print("\n" + "-"*80)
print("Test 3: Without GNN Features")
print("-"*80)

model_no_gnn = build_ablation_model(
    [X_contrastive_scaled.shape[1], X_attention_scaled.shape[1], X_structured.shape[1]],
    ['contrastive', 'attention', 'structured'],
    num_classes
)

model_no_gnn.fit(
    [X_contrastive_scaled, X_attention_scaled, X_structured],
    y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

_, acc_no_gnn = model_no_gnn.evaluate(
    [X_contrastive_scaled_test, X_attention_scaled_test, X_structured_test],
    y_test_cat,
    verbose=0
)

print(f"Without GNN: {acc_no_gnn*100:.2f}%")
print(f"Drop: {(full_acc - acc_no_gnn)*100:.2f}%")

# ==================== Test 4: Without Attention ====================
print("\n" + "-"*80)
print("Test 4: Without Attention Features")
print("-"*80)

model_no_attention = build_ablation_model(
    [X_contrastive_scaled.shape[1], X_gnn_scaled.shape[1], X_structured.shape[1]],
    ['contrastive', 'gnn', 'structured'],
    num_classes
)

model_no_attention.fit(
    [X_contrastive_scaled, X_gnn_scaled, X_structured],
    y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

_, acc_no_attention = model_no_attention.evaluate(
    [X_contrastive_scaled_test, X_gnn_scaled_test, X_structured_test],
    y_test_cat,
    verbose=0
)

print(f"Without Attention: {acc_no_attention*100:.2f}%")
print(f"Drop: {(full_acc - acc_no_attention)*100:.2f}%")

# ==================== Test 5: Without Structured ====================
print("\n" + "-"*80)
print("Test 5: Without Structured Features")
print("-"*80)

model_no_structured = build_ablation_model(
    [X_contrastive_scaled.shape[1], X_gnn_scaled.shape[1], X_attention_scaled.shape[1]],
    ['contrastive', 'gnn', 'attention'],
    num_classes
)

model_no_structured.fit(
    [X_contrastive_scaled, X_gnn_scaled, X_attention_scaled],
    y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

_, acc_no_structured = model_no_structured.evaluate(
    [X_contrastive_scaled_test, X_gnn_scaled_test, X_attention_scaled_test],
    y_test_cat,
    verbose=0
)

print(f"Without Structured: {acc_no_structured*100:.2f}%")
print(f"Drop: {(full_acc - acc_no_structured)*100:.2f}%")

# ==================== Summary ====================
print("\n" + "="*80)
print("Ablation Study Summary")
print("="*80)

results = [
    ("Full Model", full_acc, 0),
    ("Without Contrastive", acc_no_contrastive, full_acc - acc_no_contrastive),
    ("Without GNN", acc_no_gnn, full_acc - acc_no_gnn),
    ("Without Attention", acc_no_attention, full_acc - acc_no_attention),
    ("Without Structured", acc_no_structured, full_acc - acc_no_structured)
]

print(f"\n{'Configuration':<25} {'Accuracy':<12} {'Drop':<10}")
print("-"*80)
for config, acc, drop in results:
    print(f"{config:<25} {acc*100:>6.2f}%      {drop*100:>6.2f}%")

# Find most important component
drops = [(name, drop) for name, _, drop in results[1:]]
most_important = max(drops, key=lambda x: x[1])

print(f"\nMost important component: {most_important[0]} (drop: {most_important[1]*100:.2f}%)")

# Save results
with open('ablation_results.txt', 'w', encoding='utf-8') as f:
    f.write("Ablation Study Results\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'Configuration':<25} {'Accuracy':<12} {'Drop':<10}\n")
    f.write("-"*80 + "\n")
    for config, acc, drop in results:
        f.write(f"{config:<25} {acc*100:>6.2f}%      {drop*100:>6.2f}%\n")
    f.write(f"\nMost important: {most_important[0]} (drop: {most_important[1]*100:.2f}%)\n")

print("\nResults saved to 'ablation_results.txt'")


# %%



