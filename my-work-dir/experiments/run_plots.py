import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.decomposition import PCA

from utils.plotting import plot_embeddings_3d
from evaluation.linear_probe import run_linear_probe

def reduce_to_3d(embeddings):
    pca = PCA(n_components=3)
    return pca.fit_transform(embeddings)

def get_embedding(image_path, preprocess, embedding_model, vgg):
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        features = vgg.features(img_t)
        features = torch.flatten(features, 1)
        emb = embedding_model(features)

    return emb.squeeze().numpy()

def compute_embeddings(df, images_folder, preprocess, embedding_model, vgg):
    embeddings = []
    labels = []

    for _, row in df.iterrows():
        path = f"{images_folder}/{row['filename']}_CROP_SUMIMG.png"  # adjust if needed
        emb = get_embedding(path, preprocess, embedding_model, vgg)
        embeddings.append(emb)
        labels.append(row["class"])

    return np.array(embeddings), np.array(labels)

def create_vgg16_embedings_3dplot(csv_path, labels_csv, images_folder):

    # Load pretrained VGG16
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg.eval()

    # Extract the fc2 layer (second-to-last)
    embedding_model = nn.Sequential(*list(vgg.classifier.children())[:-1])

    # Preprocess as in ImageNet
    preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    ])

    df = pd.read_csv(csv_path, sep=";")
    df_labels = pd.read_csv(labels_csv, sep=";")

    final_df = pd.DataFrame(columns=df.columns)

    # Get at most 20 samples from each class
    counts = dict(df_labels['class'].value_counts())
    for label, count in counts.items():

        if label == "meteor":
            # Special case: take 20 rows directly from df
            selected_from_df = (
                df[df['class'] == label]
                .head(20)
                .copy()
            )
        else:

            selected_rows = (
                df_labels[df_labels['class'] == label]
                .head(min(count, 20))
            )
            selected_filenames = selected_rows[['filename', 'class']]
            merged = df.merge(selected_filenames, on='filename', how='inner', suffixes=('', '_new'))
            merged['class'] = merged['class_new']
            selected_from_df = merged.drop(columns=['class_new'])
        
        final_df = pd.concat([final_df, selected_from_df], ignore_index=True)

    embeddings, labels = compute_embeddings(df=final_df, images_folder=images_folder, preprocess=preprocess, embedding_model=embedding_model, vgg=vgg)
    
    embeddings_3d = reduce_to_3d(embeddings=embeddings)
    return embeddings_3d, labels

def run_plot3d_embeddings(cfg):

    images_folder = cfg['paths']['data_root']
    csv_path = cfg['paths']['dataset']
    labels_df = cfg['paths']['labels_csv']

    embeddings_3d, labels = create_vgg16_embedings_3dplot(csv_path, labels_df, images_folder)

    plot_embeddings_3d(embeddings_3d, labels)

    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    embeddings, labels = compute_embeddings(
        df=pd.read_csv(csv_path, sep=";"), 
        images_folder=images_folder, 
        preprocess= transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
            ]),
        embedding_model=nn.Sequential(*list(vgg.classifier.children())[:-1]),
        vgg=vgg)
    
    from sklearn.model_selection import train_test_split
    train_feats, test_feats, train_labels, test_labels = train_test_split(
        embeddings,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )

    val_acc, train_acc = run_linear_probe(
        train_feats=train_feats,
        train_labels=train_labels,
        val_feats=test_feats,
        val_labels=test_labels
    )

    print(f"Linear Probe — Train Acc: {train_acc:.4f}, Test Acc: {val_acc:.4f}")