import cv2
import json
import csv
import os
import yt_dlp
from datetime import datetime
from pathlib import Path

class YouTubeScreenshotCollector:
    def __init__(self, output_dir=None):
        if output_dir is None:
            # Default to data/youtube_screenshots relative to project root
            project_root = Path(__file__).parent.parent
            self.output_dir = project_root / "data" / "youtube_screenshots"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.annotations = {}
        
    def download_youtube_video(self, youtube_url, output_filename=None):
        """
        Download YouTube video and return the path to the downloaded file
        
        Args:
            youtube_url: URL of the YouTube video
            output_filename: Optional custom filename (without extension)
        
        Returns:
            str: Path to the downloaded video file
        """
        # Set up videos directory
        project_root = Path(__file__).parent.parent
        videos_dir = project_root / "data" / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]',  # Download best quality mp4
            'outtmpl': str(videos_dir / f'{output_filename or "%(title)s"}.%(ext)s'),
            'noplaylist': True,
            'cookiesfrombrowser': ('chrome',),  # Use Chrome cookies for authentication
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get info about the video
            info = ydl.extract_info(youtube_url, download=False)
            
            # Generate filename if not provided
            if not output_filename:
                # Clean up title for filename
                title = info.get('title', 'video')
                output_filename = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            # Update template with actual filename
            ydl_opts['outtmpl'] = str(videos_dir / f'{output_filename}.%(ext)s')
            
            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([youtube_url])
            
            # Return the path to downloaded file
            video_path = videos_dir / f'{output_filename}.mp4'
            print(f"Downloaded video: {video_path}")
            return str(video_path)
        
    def load_timestamps_from_csv(self, csv_path):
        """
        Load timestamps from CSV file with columns: mins, secs, play_type, description
        Returns tuple: (timestamps_list, csv_filename_without_extension)
        """
        timestamps = []
        
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Convert mins and secs to total seconds
                total_seconds = int(row['mins']) * 60 + int(row['secs'])
                
                timestamp_dict = {
                    'time': total_seconds,
                    'play_type': row['play_type'],
                    'description': row.get('description', '')
                }
                timestamps.append(timestamp_dict)
        
        # Extract filename without extension
        csv_filename = Path(csv_path).stem
        
        return timestamps, csv_filename
        
    def delete_video(self, video_path):
        """
        Delete the video file to save disk space after processing
        
        Args:
            video_path: Path to the video file to delete
            
        Returns:
            bool: True if file was deleted successfully, False otherwise
        """
        try:
            video_file = Path(video_path)
            if video_file.exists():
                video_file.unlink()
                print(f"Deleted video: {video_path}")
                return True
            else:
                print(f"Video file not found: {video_path}")
                return False
        except Exception as e:
            print(f"Error deleting video {video_path}: {e}")
            return False
        
    def process_video_timestamps(self, video_path, timestamps, csv_filename=None):
        """
        timestamps: List of dicts with keys:
        - 'time': timestamp in seconds
        - 'play_type': 'run' or 'pass'
        - 'description': optional play description
        
        Args:
            video_path: Path to the video file
            timestamps: List of timestamp dictionaries
            csv_filename: Optional name of the CSV file (without extension) for naming annotations
        """
        cap = cv2.VideoCapture(video_path)
        
        for idx, timestamp_info in enumerate(timestamps):
            # Navigate to exact timestamp
            target_time = timestamp_info['time'] * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time)
            
            ret, frame = cap.read()
            if ret:
                # Generate filename
                filename = f"{timestamp_info['play_type']}_{idx:04d}_{int(timestamp_info['time'])}.jpg"
                filepath = self.output_dir / filename
                
                # Save frame
                cv2.imwrite(str(filepath), frame)
                
                # Store annotation
                self.annotations[filename] = {
                    'play_type': timestamp_info['play_type'],
                    'description': timestamp_info.get('description', ''),
                    'source_video': video_path,
                    'timestamp': timestamp_info['time']
                }
                
                print(f"Saved: {filename}")
        
        # Save annotations with dynamic filename
        if csv_filename:
            annotations_filename = f'annotations_{csv_filename}.json'
        else:
            annotations_filename = 'annotations.json'
        
        with open(self.output_dir / annotations_filename, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        cap.release()
        return len(self.annotations)