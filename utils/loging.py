import json
import os
from datetime import datetime

def log_segment_processing(log_type: str, data: dict, log_dir: str = "output/logs"):
    """
    Save segment processing logs as JSON files.
    
    Args:
        log_type: Type of log ('ocr', 'translation', 'segment', 'frame_overlay')
        data: Dictionary containing log data
        log_dir: Directory to save logs
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create filename with timestamp and type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{log_type}_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    
    # Add metadata
    data['log_type'] = log_type
    data['timestamp'] = datetime.now().isoformat()
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"📝 Log saved: {filepath}")
    return filepath


def log_batch_summary(batch_id: int, start_frame: int, end_frame: int, 
                     segments: list, log_dir: str = "output/logs"):
    """Log summary of batch processing."""
    data = {
        'batch_id': batch_id,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'total_frames': end_frame - start_frame,
        'segments_found': len(segments),
        'segments': segments
    }
    return log_segment_processing('batch_summary', data, log_dir)