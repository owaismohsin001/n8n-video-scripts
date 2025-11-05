#!/usr/bin/env python3
"""
Log Analysis Tool for Video Processing
Analyzes JSON log files to identify bottlenecks and issues
"""

import json
import os
import sys
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any

def parse_timestamp(ts: str) -> datetime:
    """Parse ISO format timestamp"""
    return datetime.fromisoformat(ts)

def analyze_batch_status(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze status of all batches"""
    batch_info = defaultdict(lambda: {
        'started': False,
        'completed': False,
        'error': None,
        'start_time': None,
        'end_time': None,
        'duration': None,
        'events': []
    })
    
    for log in logs:
        event_type = log.get('event_type')
        data = log.get('data', {})
        batch_id = data.get('batch_id')
        timestamp = log.get('timestamp')
        
        if batch_id is not None:
            batch_info[batch_id]['events'].append({
                'event': event_type,
                'timestamp': timestamp,
                'data': data
            })
            
            if event_type == 'batch_start':
                batch_info[batch_id]['started'] = True
                batch_info[batch_id]['start_time'] = timestamp
            elif event_type == 'batch_complete':
                batch_info[batch_id]['completed'] = True
                batch_info[batch_id]['end_time'] = timestamp
                if batch_info[batch_id]['start_time']:
                    start = parse_timestamp(batch_info[batch_id]['start_time'])
                    end = parse_timestamp(timestamp)
                    batch_info[batch_id]['duration'] = (end - start).total_seconds()
            elif event_type in ['batch_error', 'batch_fatal_error']:
                batch_info[batch_id]['error'] = data.get('error')
    
    return dict(batch_info)

def analyze_thread_activity(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze thread activity and identify stuck threads"""
    thread_activity = defaultdict(lambda: {
        'last_event': None,
        'last_timestamp': None,
        'event_count': 0,
        'batch_id': None
    })
    
    for log in logs:
        thread_id = log.get('thread_id')
        thread_name = log.get('thread_name')
        timestamp = log.get('timestamp')
        event_type = log.get('event_type')
        data = log.get('data', {})
        
        if thread_id:
            key = f"{thread_id}_{thread_name}"
            thread_activity[key]['last_event'] = event_type
            thread_activity[key]['last_timestamp'] = timestamp
            thread_activity[key]['event_count'] += 1
            if 'batch_id' in data:
                thread_activity[key]['batch_id'] = data['batch_id']
    
    return dict(thread_activity)

def find_slow_operations(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find operations that took unusually long"""
    operations = []
    operation_stack = {}
    
    start_events = {
        'batch_start': 'batch_complete',
        'batch_translation_start': 'batch_translation_complete',
        'batch_overlay_start': 'batch_overlay_complete',
        'ocr_extraction_start': 'ocr_extraction_complete',
        'batch_text_change_search_start': 'batch_text_change_search_end'
    }
    
    for log in logs:
        event_type = log.get('event_type')
        timestamp = log.get('timestamp')
        data = log.get('data', {})
        batch_id = data.get('batch_id', 'unknown')
        
        # Check if this is a start event
        if event_type in start_events:
            key = f"{batch_id}_{event_type}"
            operation_stack[key] = {
                'start_event': event_type,
                'end_event': start_events[event_type],
                'start_time': timestamp,
                'batch_id': batch_id,
                'data': data
            }
        
        # Check if this is an end event
        for start_event, end_event in start_events.items():
            if event_type == end_event:
                key = f"{batch_id}_{start_event}"
                if key in operation_stack:
                    op = operation_stack[key]
                    start_time = parse_timestamp(op['start_time'])
                    end_time = parse_timestamp(timestamp)
                    duration = (end_time - start_time).total_seconds()
                    
                    operations.append({
                        'operation': start_event.replace('_start', ''),
                        'batch_id': batch_id,
                        'duration_seconds': duration,
                        'start_time': op['start_time'],
                        'end_time': timestamp
                    })
                    del operation_stack[key]
    
    # Sort by duration
    operations.sort(key=lambda x: x['duration_seconds'], reverse=True)
    return operations

def find_stuck_operations(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find operations that started but never completed"""
    stuck_operations = []
    operation_stack = {}
    
    start_events = {
        'batch_start': 'batch_complete',
        'batch_translation_start': 'batch_translation_complete',
        'batch_overlay_start': 'batch_overlay_complete',
        'ocr_extraction_start': 'ocr_extraction_complete',
        'batch_text_change_search_start': 'batch_text_change_search_end',
        'batch_no_text_scan_start': 'batch_no_text_segment_created'
    }
    
    for log in logs:
        event_type = log.get('event_type')
        timestamp = log.get('timestamp')
        data = log.get('data', {})
        batch_id = data.get('batch_id', 'unknown')
        frame = data.get('frame_index') or data.get('current_frame') or data.get('search_from_frame')
        
        # Check if this is a start event
        if event_type in start_events:
            key = f"{batch_id}_{event_type}_{frame}"
            operation_stack[key] = {
                'event': event_type,
                'batch_id': batch_id,
                'frame': frame,
                'start_time': timestamp,
                'data': data
            }
        
        # Check if this is an end event
        for start_event, end_event in start_events.items():
            if event_type == end_event:
                # Find and remove matching start event
                keys_to_remove = []
                for key, op in operation_stack.items():
                    if op['event'] == start_event and op['batch_id'] == batch_id:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del operation_stack[key]
    
    # Remaining items in stack are stuck operations
    for key, op in operation_stack.items():
        stuck_operations.append(op)
    
    return stuck_operations

def analyze_event_frequency(logs: List[Dict[str, Any]]) -> Counter:
    """Count frequency of each event type"""
    return Counter(log.get('event_type') for log in logs)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <log_file.json>")
        print("\nOr to analyze the latest log:")
        print("python analyze_logs.py latest")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # If 'latest' is specified, find the most recent log file
    if log_file == 'latest':
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            print(f"Error: logs directory not found: {log_dir}")
            sys.exit(1)
        
        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) 
                     if f.endswith('.json')]
        
        if not log_files:
            print(f"Error: No log files found in {log_dir}")
            sys.exit(1)
        
        log_file = max(log_files, key=os.path.getmtime)
        print(f"Analyzing latest log: {log_file}\n")
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    # Load logs
    print(f"Loading logs from: {log_file}")
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    print(f"Total events: {len(logs)}\n")
    
    # Event frequency
    print("="*60)
    print("EVENT FREQUENCY")
    print("="*60)
    event_freq = analyze_event_frequency(logs)
    for event, count in event_freq.most_common(15):
        print(f"{event:40s} : {count:5d}")
    print()
    
    # Batch status
    print("="*60)
    print("BATCH STATUS")
    print("="*60)
    batch_status = analyze_batch_status(logs)
    for batch_id in sorted(batch_status.keys()):
        info = batch_status[batch_id]
        status = "✅ COMPLETED" if info['completed'] else "❌ INCOMPLETE"
        if info['error']:
            status = "💥 ERROR"
        
        duration_str = f"{info['duration']:.2f}s" if info['duration'] else "N/A"
        
        print(f"Batch {batch_id}: {status} (Duration: {duration_str}, Events: {len(info['events'])})")
        if info['error']:
            print(f"  Error: {info['error']}")
    print()
    
    # Thread activity
    print("="*60)
    print("THREAD ACTIVITY")
    print("="*60)
    thread_activity = analyze_thread_activity(logs)
    for thread, info in sorted(thread_activity.items(), key=lambda x: x[1]['last_timestamp'], reverse=True)[:10]:
        batch_info = f"(Batch {info['batch_id']})" if info['batch_id'] is not None else ""
        print(f"{thread:30s} {batch_info:15s}")
        print(f"  Last Event: {info['last_event']}")
        print(f"  Last Time:  {info['last_timestamp']}")
        print(f"  Total Events: {info['event_count']}")
    print()
    
    # Slow operations
    print("="*60)
    print("TOP 10 SLOWEST OPERATIONS")
    print("="*60)
    slow_ops = find_slow_operations(logs)
    for op in slow_ops[:10]:
        print(f"{op['operation']:35s} (Batch {op['batch_id']}): {op['duration_seconds']:.2f}s")
    print()
    
    # Stuck operations
    print("="*60)
    print("STUCK/INCOMPLETE OPERATIONS")
    print("="*60)
    stuck_ops = find_stuck_operations(logs)
    if stuck_ops:
        print(f"⚠️  Found {len(stuck_ops)} operations that started but never completed!")
        for op in stuck_ops:
            frame_info = f"Frame {op['frame']}" if op['frame'] else "N/A"
            print(f"  Batch {op['batch_id']}: {op['event']:40s} ({frame_info})")
            print(f"    Started at: {op['start_time']}")
            if op['data']:
                print(f"    Data: {op['data']}")
    else:
        print("✅ No stuck operations found")
    print()
    
    # Final summary
    completed_batches = sum(1 for info in batch_status.values() if info['completed'])
    total_batches = len(batch_status)
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Batches: {total_batches}")
    print(f"Completed: {completed_batches}")
    print(f"Incomplete: {total_batches - completed_batches}")
    print(f"Stuck Operations: {len(stuck_ops)}")
    
    if logs:
        first_event = logs[0].get('timestamp')
        last_event = logs[-1].get('timestamp')
        if first_event and last_event:
            start_time = parse_timestamp(first_event)
            end_time = parse_timestamp(last_event)
            total_duration = (end_time - start_time).total_seconds()
            print(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")

if __name__ == '__main__':
    main()

