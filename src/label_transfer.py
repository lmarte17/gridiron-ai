import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

class FormationSimilarityLabeler:
    def __init__(self, similarity_threshold=0.85):
        self.threshold = similarity_threshold
        self.scaler = StandardScaler()
        self.feature_weights = {
            'personnel': 3.0,  # Personnel grouping is highly predictive
            'rb_count': 2.0,
            'formation_width': 1.5,
            'is_empty_backfield': 2.5,
            'is_heavy_formation': 2.0,
            'box_defenders': 1.0
        }
    
    def vectorize_features(self, features):
        """Convert features dict to numerical vector"""
        vector = []
        
        # Numerical features
        vector.extend([
            features.get('rb_count', 0),
            features.get('te_count', 0),
            features.get('wr_count', 0),
            features.get('formation_width', 0),
            features.get('backfield_depth', 0),
            features.get('box_defenders', 0)
        ])
        
        # Boolean features
        vector.extend([
            1 if features.get('is_empty_backfield', False) else 0,
            1 if features.get('is_heavy_formation', False) else 0,
            1 if features.get('trips_left', False) else 0,
            1 if features.get('trips_right', False) else 0,
            1 if features.get('bunch_formation', False) else 0,
            1 if features.get('compressed_set', False) else 0,
            1 if features.get('high_safety', False) else 0
        ])
        
        return np.array(vector)
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two formations"""
        # Personnel grouping match
        if features1.get('personnel') == features2.get('personnel'):
            personnel_sim = 1.0
        else:
            personnel_sim = 0.5
        
        # Vector similarity for other features
        vec1 = self.vectorize_features(features1)
        vec2 = self.vectorize_features(features2)
        
        # Normalize vectors
        if hasattr(self, 'fitted_scaler'):
            vec1 = self.fitted_scaler.transform([vec1])[0]
            vec2 = self.fitted_scaler.transform([vec2])[0]
        
        # Cosine similarity
        vector_sim = 1 - cosine(vec1, vec2)
        
        # Weighted combination
        total_sim = (personnel_sim * 0.4) + (vector_sim * 0.6)
        
        return total_sim
    
    def transfer_labels(self, labeled_features, unlabeled_features):
        """Transfer labels from labeled to unlabeled data"""
        # Fit scaler on all data
        all_vectors = []
        for features in labeled_features + unlabeled_features:
            all_vectors.append(self.vectorize_features(features['features']))
        
        self.scaler.fit(all_vectors)
        self.fitted_scaler = self.scaler
        
        # Transfer labels
        results = {}
        
        for unlabeled in unlabeled_features:
            similarities = []
            
            # Compare with all labeled examples
            for labeled in labeled_features:
                sim = self.calculate_similarity(
                    unlabeled['features'],
                    labeled['features']
                )
                similarities.append({
                    'similarity': sim,
                    'label': labeled['play_type'],
                    'source': labeled.get('filename', 'unknown')
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Determine label and confidence
            top_sim = similarities[0]['similarity']
            
            if top_sim >= self.threshold:
                # High confidence
                label = similarities[0]['label']
                confidence = 'high'
            elif top_sim >= 0.7:
                # Medium confidence - use top 3 voting
                top_labels = [s['label'] for s in similarities[:3]]
                label = max(set(top_labels), key=top_labels.count)
                confidence = 'medium'
            else:
                # Low confidence - weighted voting
                votes = {'run': 0, 'pass': 0}
                for s in similarities[:5]:
                    votes[s['label']] += s['similarity']
                label = max(votes, key=votes.get)
                confidence = 'low'
            
            results[unlabeled['filename']] = {
                'predicted_label': label,
                'confidence': confidence,
                'similarity_score': top_sim,
                'most_similar': similarities[:3]
            }
        
        return results