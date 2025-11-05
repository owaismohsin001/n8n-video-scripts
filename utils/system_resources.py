import multiprocessing
from typing import Tuple

def calculate_optimal_batch_config(total_frames: int) -> Tuple[int, int]:
    """
    Calculate optimal batch size and number of workers based on:
    - Total frames in video
    - Available CPU cores
    - Memory considerations
    
    Maximum safe batch size: 150 frames
    
    Important: Batch SIZE is NOT CPU-dependent (it's an accuracy constraint)
    - Accuracy constraint: Larger batches (>200) miss text changes at boundaries
    - This is an algorithmic limit, not a performance limit
    - Same limit applies to fast and slow CPUs
    
    What IS CPU-dependent:
    - Worker COUNT: More CPU cores = more parallel workers (line 21)
    - Processing SPEED: Faster CPU = faster batch processing
    - But safe batch SIZE remains the same regardless of CPU power
    
    Returns:
        (batch_size, num_workers)
    """
    # Get CPU count (leave 1-2 cores for system)
    cpu_count = multiprocessing.cpu_count()
    num_workers = max(1, min(cpu_count - 1, 10))  # Cap at 10 workers
    
   
    if total_frames < 500:
        batch_size = 25  # Very small videos: small batches
    elif total_frames < 10_000:
        batch_size = 100  # Medium videos: safe for accuracy
    elif total_frames < 40_000:
        batch_size = 120  # Large videos: safe for accuracy
    elif total_frames < 100_000:
        batch_size = 150  # Very large videos: maximum safe size
    else:
        batch_size = 150  # Extremely large videos: cap at safe maximum
    
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
