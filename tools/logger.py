def log_segment_duration(synthesized_audio, start_time_ms, end_time_ms, segment_id, text):
    import os

    os.makedirs('logs', exist_ok=True)

    synthesized_duration_ms = len(synthesized_audio)
    original_duration_ms = end_time_ms - start_time_ms
    duration_difference_ms = synthesized_duration_ms - original_duration_ms

    log_content = (
        f"Segment ID: {segment_id}\n"
        f"Original Duration: {original_duration_ms} ms\n"
        f"Synthesized Duration: {synthesized_duration_ms} ms\n"
        f"Duration Difference: {duration_difference_ms} ms\n"
        f"Text: {text}\n"
    )

    log_filename = f'logs/segment_duration_{segment_id}.txt'
    with open(log_filename, 'w') as log_file:
        log_file.write(log_content)
