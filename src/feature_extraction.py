import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from pathlib import Path
import yaml

class FormationFeatureExtractor:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default to fine-tuned football detection model
            project_root = Path(__file__).parent.parent
            model_path = project_root / "runs" / "detect" / "football_detector2" / "weights" / "best.pt"
            
        self.model = YOLO(str(model_path))
        
        # Load position mapping from data.yaml to ensure consistency
        project_root = Path(__file__).parent.parent
        data_yaml_path = project_root / "data" / "original" / "data.yaml"
        
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            # Create mapping from class index to class name
            self.position_mapping = {i: name for i, name in enumerate(data_config['names'])}
        else:
            # Fallback mapping based on your data.yaml
            self.position_mapping = {
                0: 'defense',
                1: 'oline',
                2: 'qb',
                3: 'ref',
                4: 'running_back',
                5: 'tight_end',
                6: 'wide_receiver'
            }
    
    def extract_features(self, image_path):
        """Extract comprehensive formation features from image"""
        # Run detection
        results = self.model(image_path, conf=0.3)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        # Parse detections
        detections = self.parse_detections(results[0])
        
        # Extract features
        features = {
            # Personnel grouping
            'personnel': self.get_personnel_grouping(detections),
            'rb_count': len(detections['running_back']),
            'te_count': len(detections['tight_end']),
            'wr_count': len(detections['wide_receiver']),
            'qb_count': len(detections['qb']),
            
            # Formation shape
            'formation_width': self.calculate_formation_width(detections),
            'backfield_depth': self.calculate_backfield_depth(detections),
            'is_empty_backfield': len(detections['running_back']) == 0,
            'is_heavy_formation': len(detections['tight_end']) >= 2,
            
            # Alignment features
            'trips_left': self.check_trips(detections, 'left'),
            'trips_right': self.check_trips(detections, 'right'),
            'bunch_formation': self.detect_bunch(detections),
            'compressed_set': self.is_compressed(detections),
            
            # Defensive alignment
            'defense_count': len(detections['defense']),
            'box_defenders': self.count_box_defenders(detections),
            'high_safety': self.detect_high_safety(detections),
            
            # Spatial encoding
            'formation_hash': self.create_formation_hash(detections)
        }
        
        # Add spatial features
        spatial_features = self.extract_spatial_features(detections)
        features.update(spatial_features)
        
        return features
    
    def parse_detections(self, result):
        """Organize detections by position type"""
        detections = defaultdict(list)
        
        for box in result.boxes:
            cls = int(box.cls)
            position = self.position_mapping.get(cls, 'unknown')
            detections[position].append({
                'bbox': box.xyxy[0].cpu().numpy(),
                'conf': float(box.conf),
                'center': self.get_center(box.xyxy[0].cpu().numpy())
            })
        
        return detections
    
    def get_center(self, bbox):
        """Get center point of bounding box"""
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    
    def calculate_formation_width(self, detections):
        """Calculate horizontal spread of offensive formation"""
        offensive_players = []
        for pos in ['oline', 'running_back', 'wide_receiver', 'tight_end', 'qb']:
            offensive_players.extend([p['center'][0] for p in detections[pos]])
        
        if len(offensive_players) < 2:
            return 0
        
        return max(offensive_players) - min(offensive_players)
    
    def calculate_backfield_depth(self, detections):
        """Calculate depth of backfield players"""
        if not detections['running_back']:
            return 0
        
        ol_y = [p['center'][1] for p in detections['oline']]
        back_y = [p['center'][1] for p in detections['running_back']]
        
        if not ol_y:
            return 0
        
        ol_line = np.mean(ol_y)
        return max(back_y) - ol_line if back_y else 0
    
    def check_trips(self, detections, side):
        """Check for trips formation (3+ receivers on one side)"""
        if len(detections['wide_receiver']) < 3:
            return False
        
        # Get field center from offensive line
        ol_centers = [p['center'][0] for p in detections['oline']]
        if not ol_centers:
            return False
        
        field_center = np.mean(ol_centers)
        
        if side == 'left':
            side_receivers = [p for p in detections['wide_receiver'] if p['center'][0] < field_center]
        else:
            side_receivers = [p for p in detections['wide_receiver'] if p['center'][0] > field_center]
        
        return len(side_receivers) >= 3
    
    def detect_bunch(self, detections):
        """Detect bunch formation (receivers close together)"""
        if len(detections['wide_receiver']) < 2:
            return False
        
        receiver_centers = [p['center'] for p in detections['wide_receiver']]
        
        # Check for receivers within close proximity
        for i, r1 in enumerate(receiver_centers):
            close_count = 1
            for j, r2 in enumerate(receiver_centers):
                if i != j:
                    distance = np.sqrt((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2)
                    if distance < 50:  # Adjust threshold as needed
                        close_count += 1
            
            if close_count >= 2:
                return True
        
        return False
    
    def is_compressed(self, detections):
        """Check if formation is compressed (narrow spread)"""
        width = self.calculate_formation_width(detections)
        return width < 200  # Adjust threshold as needed
    
    def count_box_defenders(self, detections):
        """Count defenders in the box area"""
        if not detections['oline']:
            return len(detections['defense'])
        
        # Define box area around offensive line
        ol_centers = [p['center'] for p in detections['oline']]
        ol_x = [c[0] for c in ol_centers]
        ol_y = [c[1] for c in ol_centers]
        
        box_left = min(ol_x) - 50
        box_right = max(ol_x) + 50
        box_top = min(ol_y) - 30
        box_bottom = max(ol_y) + 30
        
        box_count = 0
        for defender in detections['defense']:
            x, y = defender['center']
            if box_left <= x <= box_right and box_top <= y <= box_bottom:
                box_count += 1
        
        return box_count
    
    def detect_high_safety(self, detections):
        """Detect high safety coverage"""
        if not detections['defense']:
            return False
        
        # Find deepest defenders
        defender_depths = [p['center'][1] for p in detections['defense']]
        if len(defender_depths) < 2:
            return False
        
        deepest_defenders = sorted(defender_depths)[-2:]  # Two deepest
        depth_threshold = np.mean(defender_depths) + np.std(defender_depths)
        
        return min(deepest_defenders) > depth_threshold
    
    def create_formation_hash(self, detections):
        """Create spatial hash of formation"""
        # Simple hash based on relative positions
        offensive_positions = []
        for pos in ['oline', 'running_back', 'wide_receiver', 'tight_end', 'qb']:
            for player in detections[pos]:
                offensive_positions.append((pos, player['center'][0], player['center'][1]))
        
        # Sort by x-coordinate and create hash
        offensive_positions.sort(key=lambda x: x[1])
        hash_string = ''.join([p[0][0] for p in offensive_positions])  # First letter of position
        
        return hash_string
    
    def get_personnel_grouping(self, detections):
        """Standard NFL personnel grouping (e.g., '11', '21', '12')"""
        rb = len(detections['running_back'])
        te = len(detections['tight_end'])
        return f"{rb}{te}"
    
    def extract_spatial_features(self, detections):
        """Extract comprehensive spatial and distance-based features"""
        spatial_features = {}
        
        # Player positioning features
        spatial_features.update(self.calculate_position_features(detections))
        
        # Distance metrics
        spatial_features.update(self.calculate_distance_features(detections))
        
        # Formation geometry
        spatial_features.update(self.calculate_geometry_features(detections))
        
        # Tactical spacing
        spatial_features.update(self.calculate_tactical_features(detections))
        
        return spatial_features
    
    def calculate_position_features(self, detections):
        """Calculate positioning and spread features"""
        features = {}
        
        # Get all offensive player positions
        offensive_positions = []
        for pos in ['oline', 'running_back', 'wide_receiver', 'tight_end', 'qb']:
            for player in detections[pos]:
                offensive_positions.append(player['center'])
        
        if len(offensive_positions) >= 2:
            x_coords = [pos[0] for pos in offensive_positions]
            y_coords = [pos[1] for pos in offensive_positions]
            
            # Formation center of mass
            features['formation_center_x'] = np.mean(x_coords)
            features['formation_center_y'] = np.mean(y_coords)
            
            # Formation spread/variance
            features['formation_x_spread'] = np.std(x_coords)
            features['formation_y_spread'] = np.std(y_coords)
            
            # Formation bounding box
            features['formation_bbox_width'] = max(x_coords) - min(x_coords)
            features['formation_bbox_height'] = max(y_coords) - min(y_coords)
        else:
            features.update({
                'formation_center_x': 0, 'formation_center_y': 0,
                'formation_x_spread': 0, 'formation_y_spread': 0,
                'formation_bbox_width': 0, 'formation_bbox_height': 0
            })
        
        # Position group centers
        for pos_name in ['oline', 'wide_receiver', 'running_back', 'defense']:
            if detections[pos_name]:
                pos_centers = [p['center'] for p in detections[pos_name]]
                x_coords = [c[0] for c in pos_centers]
                y_coords = [c[1] for c in pos_centers]
                features[f'{pos_name}_center_x'] = np.mean(x_coords)
                features[f'{pos_name}_center_y'] = np.mean(y_coords)
                features[f'{pos_name}_spread_x'] = np.std(x_coords) if len(x_coords) > 1 else 0
                features[f'{pos_name}_spread_y'] = np.std(y_coords) if len(y_coords) > 1 else 0
            else:
                features.update({
                    f'{pos_name}_center_x': 0, f'{pos_name}_center_y': 0,
                    f'{pos_name}_spread_x': 0, f'{pos_name}_spread_y': 0
                })
        
        return features
    
    def calculate_distance_features(self, detections):
        """Calculate distance-based features between player groups"""
        features = {}
        
        # Average receiver spacing
        if len(detections['wide_receiver']) >= 2:
            wr_centers = [p['center'] for p in detections['wide_receiver']]
            distances = []
            for i, wr1 in enumerate(wr_centers):
                for j, wr2 in enumerate(wr_centers[i+1:], i+1):
                    dist = np.sqrt((wr1[0] - wr2[0])**2 + (wr1[1] - wr2[1])**2)
                    distances.append(dist)
            features['avg_wr_spacing'] = np.mean(distances)
            features['min_wr_spacing'] = np.min(distances)
            features['max_wr_spacing'] = np.max(distances)
        else:
            features.update({'avg_wr_spacing': 0, 'min_wr_spacing': 0, 'max_wr_spacing': 0})
        
        # QB to RB distance (pocket formation)
        if detections['qb'] and detections['running_back']:
            qb_pos = detections['qb'][0]['center']  # Assume first QB
            rb_distances = []
            for rb in detections['running_back']:
                dist = np.sqrt((qb_pos[0] - rb['center'][0])**2 + (qb_pos[1] - rb['center'][1])**2)
                rb_distances.append(dist)
            features['qb_rb_min_distance'] = np.min(rb_distances)
            features['qb_rb_avg_distance'] = np.mean(rb_distances)
        else:
            features.update({'qb_rb_min_distance': 0, 'qb_rb_avg_distance': 0})
        
        # Offensive line spacing consistency
        if len(detections['oline']) >= 2:
            ol_centers = [p['center'] for p in detections['oline']]
            ol_centers.sort(key=lambda x: x[0])  # Sort by x-coordinate
            spacings = []
            for i in range(len(ol_centers) - 1):
                spacing = abs(ol_centers[i+1][0] - ol_centers[i][0])
                spacings.append(spacing)
            features['oline_avg_spacing'] = np.mean(spacings)
            features['oline_spacing_consistency'] = 1 / (1 + np.std(spacings))  # Higher = more consistent
        else:
            features.update({'oline_avg_spacing': 0, 'oline_spacing_consistency': 0})
        
        # Nearest defender distances for offensive players
        if detections['defense']:
            def_centers = [p['center'] for p in detections['defense']]
            
            for off_pos in ['qb', 'running_back', 'wide_receiver']:
                if detections[off_pos]:
                    min_distances = []
                    for off_player in detections[off_pos]:
                        distances_to_def = []
                        for def_center in def_centers:
                            dist = np.sqrt((off_player['center'][0] - def_center[0])**2 + 
                                         (off_player['center'][1] - def_center[1])**2)
                            distances_to_def.append(dist)
                        min_distances.append(min(distances_to_def))
                    
                    features[f'{off_pos}_min_def_distance'] = np.min(min_distances)
                    features[f'{off_pos}_avg_def_distance'] = np.mean(min_distances)
                else:
                    features.update({
                        f'{off_pos}_min_def_distance': 0,
                        f'{off_pos}_avg_def_distance': 0
                    })
        
        return features
    
    def calculate_geometry_features(self, detections):
        """Calculate formation geometry and alignment features"""
        features = {}
        
        # Formation symmetry (left vs right balance)
        if detections['oline']:
            ol_center_x = np.mean([p['center'][0] for p in detections['oline']])
            
            # Count players on each side
            left_players = 0
            right_players = 0
            
            for pos in ['wide_receiver', 'tight_end', 'running_back']:
                for player in detections[pos]:
                    if player['center'][0] < ol_center_x:
                        left_players += 1
                    else:
                        right_players += 1
            
            total_skill = left_players + right_players
            if total_skill > 0:
                features['formation_symmetry'] = 1 - abs(left_players - right_players) / total_skill
            else:
                features['formation_symmetry'] = 1
        else:
            features['formation_symmetry'] = 0
        
        # Alignment quality (how well players line up)
        if len(detections['oline']) >= 3:
            ol_y_coords = [p['center'][1] for p in detections['oline']]
            features['oline_alignment_quality'] = 1 / (1 + np.std(ol_y_coords))
        else:
            features['oline_alignment_quality'] = 0
        
        # Depth staggering (layered formation)
        all_offensive = []
        for pos in ['oline', 'running_back', 'wide_receiver', 'tight_end', 'qb']:
            for player in detections[pos]:
                all_offensive.append(player['center'][1])  # y-coordinate
        
        if len(all_offensive) >= 3:
            features['depth_layers'] = len(set(np.round(all_offensive, -1)))  # Round to nearest 10 pixels
            features['depth_variance'] = np.var(all_offensive)
        else:
            features.update({'depth_layers': 0, 'depth_variance': 0})
        
        return features
    
    def calculate_tactical_features(self, detections):
        """Calculate tactical spacing and leverage features"""
        features = {}
        
        # Leverage ratios (offense vs defense positioning)
        if detections['defense'] and detections['oline']:
            def_center_x = np.mean([p['center'][0] for p in detections['defense']])
            off_center_x = np.mean([p['center'][0] for p in detections['oline']])
            features['horizontal_leverage'] = abs(def_center_x - off_center_x)
            
            def_center_y = np.mean([p['center'][1] for p in detections['defense']])
            off_center_y = np.mean([p['center'][1] for p in detections['oline']])
            features['vertical_leverage'] = def_center_y - off_center_y  # Positive = defense deeper
        else:
            features.update({'horizontal_leverage': 0, 'vertical_leverage': 0})
        
        # Coverage gaps (spaces between defenders)
        if len(detections['defense']) >= 2:
            def_x_coords = sorted([p['center'][0] for p in detections['defense']])
            gaps = []
            for i in range(len(def_x_coords) - 1):
                gap = def_x_coords[i+1] - def_x_coords[i]
                gaps.append(gap)
            features['avg_coverage_gap'] = np.mean(gaps)
            features['max_coverage_gap'] = np.max(gaps)
        else:
            features.update({'avg_coverage_gap': 0, 'max_coverage_gap': 0})
        
        # Pressure angles (how defense is positioned relative to QB)
        if detections['qb'] and detections['defense']:
            qb_pos = detections['qb'][0]['center']
            pressure_angles = []
            
            for defender in detections['defense']:
                # Calculate angle from QB to defender
                dx = defender['center'][0] - qb_pos[0]
                dy = defender['center'][1] - qb_pos[1]
                angle = np.arctan2(dy, dx)
                pressure_angles.append(angle)
            
            features['pressure_angle_spread'] = np.std(pressure_angles)
            features['avg_pressure_angle'] = np.mean(pressure_angles)
        else:
            features.update({'pressure_angle_spread': 0, 'avg_pressure_angle': 0})
        
        # Formation density (players per unit area)
        all_players = []
        for pos in ['oline', 'running_back', 'wide_receiver', 'tight_end', 'qb', 'defense']:
            for player in detections[pos]:
                all_players.append(player['center'])
        
        if len(all_players) >= 3:
            x_coords = [p[0] for p in all_players]
            y_coords = [p[1] for p in all_players]
            area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
            features['player_density'] = len(all_players) / (area + 1)  # +1 to avoid division by zero
        else:
            features['player_density'] = 0
        
        return features