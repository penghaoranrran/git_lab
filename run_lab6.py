import pandas as pd
import numpy as np
from decision_tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDT
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data(path):
    try:
        return pd.read_csv(path, encoding='gbk')
    except:
        return pd.read_csv(path, encoding='utf-8')

# --- Basic Requirement: ID3 on Watermelon 1 ---
print("=== Basic Requirement: ID3 on Watermelon Dataset 1 ===")
df1 = load_data('Watermelon-train1.csv')
X1 = df1.drop(columns=['编号', '好瓜'])
y1 = df1['好瓜']

dt_id3 = DecisionTreeClassifier(criterion='id3')
dt_id3.fit(X1, y1)
print("ID3 Tree:")
dt_id3.print_tree()
acc1 = accuracy_score(y1, dt_id3.predict(X1))
print(f"Training Accuracy: {acc1:.4f}")

# --- Intermediate Requirement: Continuous & Pruning on Watermelon 2 ---
print("\n=== Intermediate Requirement: Continuous Features & Pruning on Watermelon Dataset 2 ===")
df2 = load_data('Watermelon-train2.csv')
target_col = [c for c in df2.columns if '好' in c and '瓜' in c][0]
X2 = df2.drop(columns=['编号', target_col])
y2 = df2[target_col]

# Split for Post-pruning
X_train, X_val, y_train, y_val = train_test_split(X2, y2, test_size=0.3, random_state=42)

print("\n--- Pre-pruning (Max Depth=2) ---")
dt_pre = DecisionTreeClassifier(criterion='id3', max_depth=2)
dt_pre.fit(X_train, y_train)
acc_pre = accuracy_score(y_val, dt_pre.predict(X_val))
print(f"Validation Accuracy (Pre-pruning): {acc_pre:.4f}")

print("\n--- Post-pruning ---")
dt_post = DecisionTreeClassifier(criterion='id3', pruning='post')
dt_post.fit(X_train, y_train, X_val, y_val)
acc_post = accuracy_score(y_val, dt_post.predict(X_val))
print(f"Validation Accuracy (Post-pruning): {acc_post:.4f}")

print("\n--- Feature Importance ---")
# Train on full dataset for importance
dt_full = DecisionTreeClassifier(criterion='id3')
dt_full.fit(X2, y2)
importances = dt_full.feature_importances_
for f, imp in importances.items():
    print(f"{f}: {imp:.4f}")

# --- Advanced Requirement: C4.5 and CART ---
print("\n=== Advanced Requirement: C4.5 and CART Variants ===")
print("\n--- C4.5 (Gain Ratio) ---")
dt_c45 = DecisionTreeClassifier(criterion='c45')
dt_c45.fit(X_train, y_train)
acc_c45 = accuracy_score(y_val, dt_c45.predict(X_val))
print(f"Validation Accuracy (C4.5): {acc_c45:.4f}")

print("\n--- CART (Gini) ---")
dt_cart = DecisionTreeClassifier(criterion='cart')
dt_cart.fit(X_train, y_train)
acc_cart = accuracy_score(y_val, dt_cart.predict(X_val))
print(f"Validation Accuracy (CART): {acc_cart:.4f}")

print("\n--- Comparison with Scikit-learn ---")
# Preprocess data for sklearn (needs numeric inputs)
X2_encoded = X2.copy()
for col in X2_encoded.columns:
    if X2_encoded[col].dtype == 'object':
        X2_encoded[col] = X2_encoded[col].astype('category').cat.codes
y2_encoded = y2.astype('category').cat.codes

X_train_sk, X_val_sk, y_train_sk, y_val_sk = train_test_split(X2_encoded, y2_encoded, test_size=0.3, random_state=42)

sk_dt = SklearnDT(criterion='gini', random_state=42)
sk_dt.fit(X_train_sk, y_train_sk)
acc_sk = accuracy_score(y_val_sk, sk_dt.predict(X_val_sk))
print(f"Sklearn CART Validation Accuracy: {acc_sk:.4f}")

# --- Visualization ---
print("\nGenerating Visualizations...")

# 1. Feature Importance Plot
plt.figure(figsize=(10, 6))
features = list(importances.keys())
values = list(importances.values())

# Sort by importance
sorted_idx = np.argsort(values)
features = [features[i] for i in sorted_idx]
values = [values[i] for i in sorted_idx]

plt.barh(features, values, color='skyblue')
plt.xlabel('Information Gain (Importance)')
plt.title('Feature Importance (Watermelon Dataset 2.0)')

# Add value labels
for i, v in enumerate(values):
    plt.text(v, i, f' {v:.4f}', va='center')
    
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Saved feature_importance.png")

# 2. Model Comparison Plot
models = ['ID3 (Pre-prune)', 'ID3 (Post-prune)', 'C4.5', 'CART', 'Sklearn']
accuracies = [acc_pre, acc_post, acc_c45, acc_cart, acc_sk]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
plt.ylim(0, 1.0)
plt.ylabel('Validation Accuracy')
plt.title('Model Comparison on Watermelon Dataset 2.0')

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom')
            
plt.tight_layout()
plt.savefig('model_comparison.png')
print("Saved model_comparison.png")

# 3. Decision Boundary Visualization (Using Density and Texture)
print("Generating Decision Boundary Plot...")
# Select features: Density (Continuous) and Texture (Discrete -> Numeric)
feature_cols = ['密度', '纹理']
X_vis = df2[feature_cols].copy()
y_vis = df2[target_col]

# Encode '纹理' to numeric for visualization
# 纹理: 清晰->1, 稍糊->2, 模糊->3
texture_map = {'清晰': 1, '稍糊': 2, '模糊': 3}
X_vis['纹理'] = X_vis['纹理'].map(texture_map)

# Train a new simple tree on just these two features for visualization purposes
dt_vis = DecisionTreeClassifier(criterion='cart', max_depth=3) 
dt_vis.fit(X_vis, y_vis)

# Create meshgrid
x_min, x_max = X_vis.iloc[:, 0].min() - 0.1, X_vis.iloc[:, 0].max() + 0.1
y_min, y_max = X_vis.iloc[:, 1].min() - 0.5, X_vis.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005),
                     np.arange(y_min, y_max, 0.05))

# Predict for each point in meshgrid
mesh_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_cols)
Z = dt_vis.predict(mesh_data)

# Convert string labels to numeric for plotting contours
label_map = {label: i for i, label in enumerate(np.unique(y_vis))}
Z_num = Z.map(label_map).values.reshape(xx.shape)

plt.figure(figsize=(10, 8))
# Plot decision boundary
plt.contourf(xx, yy, Z_num, alpha=0.4, cmap=plt.cm.RdYlBu)

# Plot training points
y_vis_num = y_vis.map(label_map)
scatter = plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_vis_num, 
                      s=40, edgecolor='k', cmap=plt.cm.RdYlBu)
plt.xlabel('密度')
plt.ylabel('纹理 (1=清晰, 2=稍糊, 3=模糊)')
plt.title('Decision Tree Boundary (Density vs Texture)')
plt.legend(handles=scatter.legend_elements()[0], labels=label_map.keys())
plt.yticks([1, 2, 3], ['清晰', '稍糊', '模糊'])
plt.tight_layout()
plt.savefig('decision_boundary.png')
print("Saved decision_boundary.png")

# 4. Confusion Matrix Visualization
print("Generating Confusion Matrix Plot...")
y_pred_post = dt_post.predict(X_val)
cm = confusion_matrix(y_val, y_pred_post)
labels = sorted(list(set(y_val)))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Post-pruning Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")
