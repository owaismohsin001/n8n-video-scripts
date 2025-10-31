import multiprocessing
from typing import Tuple

def calculate_optimal_batch_config(total_frames: int) -> Tuple[int, int]:
    """
    Calculate optimal batch size and number of workers based on:
    - Total frames in video
    - Available CPU cores
    - Memory considerations
    
    Returns:
        (batch_size, num_workers)
    """
    # Get CPU count (leave 1-2 cores for system)
    cpu_count = multiprocessing.cpu_count()
    num_workers = max(1, min(cpu_count - 1, 10))  # Cap at 10 workers
    
    # Dynamic batch sizing based on video length
    if total_frames < 5_000:
        batch_size = 25  # Small videos: smaller batches
    elif total_frames < 10_000:
        batch_size = 50  # Small videos: smaller batches
    elif total_frames < 20_000:
        batch_size = 200  # Medium videos
    elif total_frames < 40_000:
        batch_size = 400  # Large videos
    else:
        batch_size = 600  # Very large videos
    
    # Ensure we have enough batches to utilize workers
    num_batches = (total_frames + batch_size - 1) // batch_size
    if num_batches < num_workers:
        # Reduce batch size to create more batches
        batch_size = max(50, total_frames // (num_workers * 2))
    
    print(f"📊 Batch Configuration:")
    print(f"   Total frames: {total_frames}")
    print(f"   Batch size: {batch_size} frames per batch")
    print(f"   Number of batches: {(total_frames + batch_size - 1) // batch_size}")
    print(f"   Workers: {num_workers} parallel threads")
    print(f"   CPU cores available: {cpu_count}")
    
    return batch_size, num_workers
