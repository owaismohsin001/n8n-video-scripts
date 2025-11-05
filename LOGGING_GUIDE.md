# Video Processing Logging System

## Overview

This project now includes comprehensive JSON-based logging to help diagnose issues with video processing, especially when batches get stuck or hang during parallel processing.

## Features

- **Thread-safe logging**: All events are logged with thread information
- **Detailed tracking**: Logs every major operation including:
  - Batch processing lifecycle
  - OCR operations
  - Text translation
  - Frame processing
  - Video writing operations
  - Thread pool management
- **JSON format**: Easy to parse and analyze programmatically
- **Automatic log files**: Logs are saved in `logs/` directory with timestamps

## Log File Location

Logs are automatically saved to:
```
logs/video_processing_YYYYMMDD_HHMMSS.json
```

Example: `logs/video_processing_20231105_143022.json`

## Using the Log Analyzer

A Python script is provided to analyze the logs and identify issues:

### Basic Usage

```bash
# Analyze a specific log file
python analyze_logs.py logs/video_processing_20231105_143022.json

# Analyze the most recent log file
python analyze_logs.py latest
```

### What the Analyzer Shows

1. **Event Frequency**: How many times each type of event occurred
2. **Batch Status**: Which batches completed, failed, or got stuck
3. **Thread Activity**: What each thread was doing last
4. **Slow Operations**: Operations that took the longest time
5. **Stuck Operations**: Operations that started but never finished (⚠️ THIS IS THE KEY!)

## Common Issues and Solutions

### Issue 1: Batches Stop Processing After Some Complete

**Symptoms**: 
- Console shows batches 1, 2, 3, 5 completed
- Batch 4 (or others) never complete
- Process hangs indefinitely

**How to Diagnose**:
```bash
python analyze_logs.py latest
```

Look for:
- "STUCK/INCOMPLETE OPERATIONS" section
- Batches marked as "❌ INCOMPLETE"
- Last events for stuck threads

**Common Causes**:
1. **OCR Hanging**: Check if `ocr_extraction_start` has no matching `ocr_extraction_complete`
   - Solution: The specific frame might be corrupted or causing OCR issues
   
2. **Translation API Timeout**: Check if `batch_line_translation_start` has no matching `batch_line_translation_end`
   - Solution: Translation service might be rate-limiting or timing out
   
3. **Video Writer Issues**: Check if `batch_video_writer_init` succeeded
   - Solution: Disk space or codec issues

4. **Text Change Search Hanging**: Check if `batch_text_change_search_start` has no matching end event
   - Solution: Exponential search might be scanning too many frames

### Issue 2: One Batch Takes Very Long

**How to Diagnose**:
Check the "TOP 10 SLOWEST OPERATIONS" section in the analyzer output.

**Interpretation**:
- `batch_translation` taking long: Many text segments or slow translation API
- `batch_overlay` taking long: Many frames to process
- `batch_text_change_search` taking long: Large gaps between text changes

### Issue 3: Thread Pool Issues

**Symptoms**:
- Some batches never start
- Thread count doesn't match expected workers

**How to Diagnose**:
Look for these events in order:
1. `thread_pool_created` - Should show correct worker count
2. `batch_submitted` - All batches should be submitted
3. `batch_start` - Each batch should actually start

## Key Events to Watch

### Critical Start/End Pairs

These events should always have matching pairs:

| Start Event | End Event | What It Tracks |
|------------|-----------|----------------|
| `batch_start` | `batch_complete` | Entire batch processing |
| `ocr_extraction_start` | `ocr_extraction_complete` | OCR on a frame |
| `batch_translation_start` | `batch_translation_complete` | Translating text |
| `batch_overlay_start` | `batch_overlay_complete` | Overlaying translations |
| `batch_text_change_search_start` | `batch_text_change_search_end` | Finding text changes |
| `batch_no_text_scan_start` | `batch_no_text_segment_created` | Scanning frames without text |

If you see a start event without a matching end event, that operation is **stuck**.

## Manual Log Analysis

If you prefer to analyze logs manually, here's the structure:

```json
{
  "timestamp": "2023-11-05T14:30:22.123456",
  "thread_id": 12345,
  "thread_name": "ThreadPoolExecutor-0_0",
  "event_type": "batch_start",
  "data": {
    "batch_id": 0,
    "start_frame": 0,
    "end_frame": 500,
    "frame_count": 500
  }
}
```

### Useful Commands

```bash
# Count events by type
cat logs/video_processing_*.json | jq '.[].event_type' | sort | uniq -c

# Find all batch_start events
cat logs/video_processing_*.json | jq '.[] | select(.event_type == "batch_start")'

# Find stuck batches (have start but no complete)
cat logs/video_processing_*.json | jq '.[] | select(.event_type == "batch_start" or .event_type == "batch_complete") | {batch_id: .data.batch_id, event: .event_type}'

# Check last 10 events
cat logs/video_processing_*.json | jq '.[-10:]'
```

## Performance Monitoring

### Frame Processing Rate

Look for `batch_frame_processing` events to see how fast frames are being processed:

```bash
cat logs/video_processing_*.json | jq '.[] | select(.event_type == "batch_frame_processing")'
```

### OCR Performance

Check OCR timing:

```bash
cat logs/video_processing_*.json | jq '.[] | select(.event_type == "ocr_extraction_complete") | {frame: .data.frame_index, time: .data.ocr_time_seconds}'
```

### Translation Performance

Check translation timing by looking at time between `batch_line_translation_start` and `batch_line_translation_end` events.

## Reducing Log Size

If log files get too large, you can adjust the logging frequency in the code:

- Frame-level logging happens every 20th frame (configurable)
- OCR cache hits are not logged (only misses)
- Regular progress updates happen every 10-20 frames

## Getting Help

When reporting issues, please:

1. Run the log analyzer: `python analyze_logs.py latest`
2. Share the analyzer output
3. If needed, share the last 50-100 events from the log file
4. Note which batch number(s) are stuck
5. Check available disk space and memory

## Example Debugging Session

```bash
# 1. Run your video processing
python main.py --video input.mp4 --targetLang English

# 2. If it hangs, open another terminal and analyze
python analyze_logs.py latest

# 3. Look for stuck operations
# Example output:
# ⚠️  Found 1 operations that started but never completed!
#   Batch 4: batch_text_change_search_start (Frame 2341)
#     Started at: 2023-11-05T14:35:22.123456

# 4. This tells us batch 4 got stuck searching for text changes at frame 2341
# Possible issues:
# - Frame 2341 might be corrupted
# - Too many frames with similar text
# - OCR hanging on specific frame content
```

## Additional Monitoring

You can also monitor:

- **CPU usage**: `top` or `htop` to see if threads are actively working
- **Memory usage**: Check if process is running out of memory
- **Disk I/O**: Check if disk writes are slow
- **Network**: If using cloud translation API, check network activity

## Future Improvements

Potential additions:
- Real-time log streaming dashboard
- Automatic stuck thread detection and recovery
- Performance benchmarking and comparison
- Log rotation for long-running processes

