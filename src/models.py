import torch
import torch.nn as nn
import torchvision.models as models
import timm

class MultiModalPlayPredictor(nn.Module):
    def __init__(self, num_formation_features=13, num_classes=2):
        super().__init__()
        
        # Visual feature extractor (CNN)
        self.visual_backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=True, 
            num_classes=0  # Remove classifier head
        )
        visual_feature_dim = self.visual_backbone.num_features  # 1280 for efficientnet_b0
        
        # Formation feature encoder
        self.formation_encoder = nn.Sequential(
            nn.Linear(num_formation_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Visual feature projection for attention
        self.visual_projection = nn.Linear(visual_feature_dim, 128)
        
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(visual_feature_dim + 128, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Auxiliary heads for multi-task learning
        self.personnel_predictor = nn.Linear(visual_feature_dim, 5)  # Predict RB count
        self.formation_width_predictor = nn.Linear(visual_feature_dim, 1)  # Predict width
        
    def forward(self, image, formation_features):
        # Extract visual features
        visual_features = self.visual_backbone(image)  # [batch, 1280]
        
        # Encode formation features
        formation_encoded = self.formation_encoder(formation_features)  # [batch, 128]
        
        # Apply attention between visual and formation features
        # Project visual features to same dimension as formation features
        visual_projected = self.visual_projection(visual_features)  # [batch, 128]
        
        # Reshape for attention layer
        visual_attn = visual_projected.unsqueeze(1)  # [batch, 1, 128]
        formation_attn = formation_encoded.unsqueeze(1)  # [batch, 1, 128]
        
        # Apply attention
        attended_formation, _ = self.attention(
            formation_attn, 
            visual_attn, 
            visual_attn
        )
        attended_formation = attended_formation.squeeze(1)  # [batch, 128]
        
        # Concatenate all features
        combined_features = torch.cat([visual_features, attended_formation], dim=1)
        
        # Main prediction
        play_prediction = self.classifier(combined_features)
        
        # Auxiliary predictions
        personnel_pred = self.personnel_predictor(visual_features)
        width_pred = self.formation_width_predictor(visual_features)
        
        return {
            'play_type': play_prediction,
            'personnel': personnel_pred,
            'width': width_pred
        }