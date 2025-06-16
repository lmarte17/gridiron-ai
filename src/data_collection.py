import cv2
import json
import csv
import os
import yt_dlp
from datetime import datetime
from pathlib import Path

class YouTubeScreenshotCollector:
    def __init__(self, output_dir="youtube_screenshots"):
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
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]',  # Download best quality mp4
            'outtmpl': str(self.output_dir / f'{output_filename or "%(title)s"}.%(ext)s'),
            'noplaylist': True,
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
            ydl_opts['outtmpl'] = str(self.output_dir / f'{output_filename}.%(ext)s')
            
            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([youtube_url])
            
            # Return the path to downloaded file
            video_path = self.output_dir / f'{output_filename}.mp4'
            print(f"Downloaded video: {video_path}")
            return str(video_path)
        
    def load_timestamps_from_csv(self, csv_path):
        """
        Load timestamps from CSV file with columns: mins, secs, play_type, description
        Returns list of dicts compatible with process_video_timestamps method
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
        
        return timestamps
        
    def process_video_timestamps(self, video_path, timestamps):
        """
        timestamps: List of dicts with keys:
        - 'time': timestamp in seconds
        - 'play_type': 'run' or 'pass'
        - 'description': optional play description
        """
        cap = cv2.VideoCapture(video_path)
        
        for idx, timestamp_info in enumerate(timestamps):
            # Navigate to timestamp (1 second before snap)
            target_time = (timestamp_info['time'] - 1.0) * 1000
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
        
        # Save annotations
        with open(self.output_dir / 'annotations.json', 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        cap.release()
        return len(self.annotations)